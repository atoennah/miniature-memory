import json
import os
from scraper.search.discover import discover_urls

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
