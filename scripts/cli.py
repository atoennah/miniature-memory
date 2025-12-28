import argparse
import json
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.search.discover import discover_urls

def main():
    parser = argparse.ArgumentParser(description="A CLI for the auto-scraping and training pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search command
    search_parser = subparsers.add_parser("search", help="Discover URLs for a given query.")
    search_parser.add_argument("query", type=str, help="The search query.")
    search_parser.add_argument("--num-results", type=int, default=10, help="The number of results to return.")

    args = parser.parse_args()

    if args.command == "search":
        print(f"Discovering URLs for query: '{args.query}'...")
        urls = discover_urls(args.query, num_results=args.num_results)

        output_file = "dataset/metadata/urls.jsonl"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "a") as f:
            for url_data in urls:
                f.write(json.dumps(url_data) + "\n")

        print(f"Successfully discovered and saved {len(urls)} URLs to {output_file}")

if __name__ == "__main__":
    main()
