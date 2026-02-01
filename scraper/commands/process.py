import json
import os
import tempfile
from playwright.sync_api import sync_playwright
from scraper.process import extract_text, BrowserManager
from scraper.storage import save_raw_text
from scraper.crawler import find_story_urls_heuristically

def process_story(page, story_url):
    """
    Processes a single story URL: navigates to the page, extracts text,
    and saves it to the raw storage.
    """
    try:
        print(f"  -> Processing story: {story_url}")
        page.goto(story_url, timeout=60000)
        html = page.content()
        text = extract_text(html)
        if text:
            save_raw_text(story_url, text)
            return True
        else:
            print(f"    -> No text extracted from {story_url}")
            return False
    except Exception as e:
        print(f"    -> Failed to process {story_url}: {e}")
        return False

def process_index_page(page, url_data):
    """
    Processes an index page: finds potential story URLs and iterates
    through them to extract content.
    """
    index_url = url_data['url']
    print(f"Processing index URL: {index_url}")

    try:
        page.goto(index_url, timeout=60000)
        # Wait for a bit to allow dynamic content/scripts to settle
        page.wait_for_timeout(5000)

        story_urls = find_story_urls_heuristically(page, index_url)

        if not story_urls:
            print(f"  -> No stories found on {index_url}. Marking as rejected.")
            url_data["status"] = "rejected"
        else:
            print(f"  -> Found {len(story_urls)} stories. Starting extraction...")
            for story_url in story_urls:
                process_story(page, story_url)
            url_data["status"] = "crawled"

    except Exception as e:
        print(f"  -> Error processing index {index_url}: {e}")
        url_data["status"] = "failed"

    return url_data

def run_process(args):
    """
    The main entry point for the 'process' command.
    Iterates through 'new' URLs in the manifest and processes them.
    """
    manifest_file = "dataset/metadata/urls.jsonl"
    if not os.path.exists(manifest_file):
        print("URL manifest file not found. Run 'search' first.")
        return

    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(manifest_file))

    try:
        with sync_playwright() as p:
            # Use BrowserManager to maintain a single browser instance
            browser_manager = BrowserManager(p)
            page = browser_manager.get_page()

            with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            url_data = json.loads(line)
                            if url_data.get("status") == "new":
                                url_data = process_index_page(page, url_data)

                            temp_f.write(json.dumps(url_data) + "\n")
                        except json.JSONDecodeError:
                            temp_f.write(line)
                            continue

            browser_manager.close()

        # Atomically update the manifest file
        os.replace(temp_path, manifest_file)
        print("\nProcessing complete. Manifest updated.")

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"An error occurred during processing: {e}")
        raise
