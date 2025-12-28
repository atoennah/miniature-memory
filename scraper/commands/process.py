import json
import os
import tempfile
from scraper.process import fetch_html, extract_text
from scraper.storage import save_raw_text

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
