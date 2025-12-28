import argparse
import json
import os
import tempfile

# The project is now structured as a package, so we can use standard imports
from scraper.search.discover import discover_urls
from scraper.process import fetch_html, extract_text
from scraper.storage import save_raw_text

def run_search(args):
    """Runs the URL discovery process."""
    print(f"Discovering URLs for query: '{args.query}'...")
    urls = discover_urls(args.query, num_results=args.num_results)

    output_file = "dataset/metadata/urls.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for url_data in urls:
            f.write(json.dumps(url_data) + "\n")

    print(f"Successfully discovered and saved {len(urls)} URLs to {output_file}")

def run_process(args):
    """Runs the fetch, extract, and save pipeline using an atomic write pattern."""
    manifest_file = "dataset/metadata/urls.jsonl"
    if not os.path.exists(manifest_file):
        print("URL manifest file not found. Run 'search' first.")
        return

    # Use a temporary file to prevent data corruption
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(manifest_file))

    try:
        with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        url_data = json.loads(line)
                        if url_data.get("status") == "new":
                            print(f"Processing URL: {url_data['url']}")
                            html = fetch_html(url_data['url'])

                            if html:
                                text = extract_text(html)
                                filepath = save_raw_text(url_data['url'], text)
                                if filepath:
                                    print(f"  -> Saved to {filepath}")
                                    url_data["status"] = "fetched"
                                else:
                                    print("  -> Rejected (empty content or save error)")
                                    url_data["status"] = "rejected"
                            else:
                                print("  -> Rejected (fetch failed)")
                                url_data["status"] = "rejected"

                        # Write the (potentially updated) line to the temporary file
                        temp_f.write(json.dumps(url_data) + "\n")

                    except json.JSONDecodeError:
                        # If a line is malformed, preserve it in the new file
                        print(f"Skipping malformed line: {line.strip()}")
                        temp_f.write(line)
                        continue

        # Atomically replace the original file with the temporary one
        os.replace(temp_path, manifest_file)

    except Exception as e:
        # Ensure the temporary file is cleaned up on error
        print(f"An error occurred during processing: {e}")
        os.remove(temp_path)
        # Re-raise the exception to make the failure clear
        raise

    print("\nProcessing complete.")

def main():
    parser = argparse.ArgumentParser(description="A CLI for the auto-scraping and training pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search command
    search_parser = subparsers.add_parser("search", help="Discover URLs for a given query.")
    search_parser.add_argument("query", type=str, help="The search query.")
    search_parser.add_argument("--num-results", type=int, default=10, help="The number of results to return.")
    search_parser.set_defaults(func=run_search)

    # Process command
    process_parser = subparsers.add_parser("process", help="Fetch, extract, and save content from new URLs.")
    process_parser.set_defaults(func=run_process)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)

if __name__ == "__main__":
    main()
