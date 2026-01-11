import json
import os
from scraper.search.discover import discover_urls
from typing import Set

def load_existing_urls(filepath: str) -> Set[str]:
    """Loads existing URLs from the manifest file into a set for quick lookup."""
    if not os.path.exists(filepath):
        return set()

    existing_urls = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                url_data = json.loads(line)
                existing_urls.add(url_data["url"])
            except (json.JSONDecodeError, KeyError):
                # Ignore malformed lines or lines without a URL key
                continue
    return existing_urls

def run_search(args):
    """
    Runs the URL discovery process and appends new, unique URLs to the manifest.
    """
    manifest_file = "dataset/metadata/urls.jsonl"
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)

    # Step 1: Load existing URLs to prevent duplicates
    existing_urls = load_existing_urls(manifest_file)
    print(f"Found {len(existing_urls)} existing URLs in the manifest.")

    # Step 2: Discover new URLs
    print(f"Discovering new URLs for query: '{args.query}'...")
    discovered_urls = discover_urls(args.query, num_results=args.num_results)

    # Step 3: Filter out URLs that already exist in the manifest
    new_urls = [
        url_data for url_data in discovered_urls
        if url_data["url"] not in existing_urls
    ]

    # Step 4: Append only the new, unique URLs to the manifest
    if new_urls:
        with open(manifest_file, "a", encoding="utf-8") as f:
            for url_data in new_urls:
                f.write(json.dumps(url_data) + "\n")
        print(f"Successfully discovered and saved {len(new_urls)} new URLs.")
    else:
        print("No new URLs were discovered.")

    print("Search complete.")
