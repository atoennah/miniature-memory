import json
import os
import tempfile
from playwright.sync_api import sync_playwright
from scraper.process import extract_text
from scraper.storage import save_raw_text
from scraper.crawler import find_story_urls_heuristically

def run_process(args):
    """
    Runs the heuristic crawl, fetch, extract, and save pipeline.
    This function treats URLs from the manifest as index pages, heuristically
    finds potential story links, and then processes them.
    """
    manifest_file = "dataset/metadata/urls.jsonl"
    if not os.path.exists(manifest_file):
        print("URL manifest file not found. Run 'search' first.")
        return

    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(manifest_file))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            url_data = json.loads(line)
                            if url_data.get("status") == "new":
                                index_url = url_data['url']

                                print(f"Processing index URL: {index_url}")
                                page.goto(index_url, timeout=60000)
                                page.wait_for_timeout(5000) # Wait for JS to settle

                                # --- Debug Snapshot ---
                                page.screenshot(path="debug_view.png")
                                with open("debug_dom.html", "w", encoding="utf-8") as f:
                                    f.write(page.content())
                                print("  -> Saved debug snapshot and DOM dump.")
                                # --- End Debug ---

                                story_urls = find_story_urls_heuristically(page, index_url)

                                if not story_urls:
                                    url_data["status"] = "rejected"
                                else:
                                    for story_url in story_urls:
                                        try:
                                            print(f"  -> Processing story: {story_url}")
                                            page.goto(story_url, timeout=60000)
                                            html = page.content()
                                            text = extract_text(html)
                                            save_raw_text(story_url, text)
                                        except Exception as e:
                                            print(f"    -> Failed to process {story_url}: {e}")
                                            continue

                                    url_data["status"] = "crawled"

                            temp_f.write(json.dumps(url_data) + "\n")

                        except json.JSONDecodeError:
                            temp_f.write(line)
                            continue

            os.replace(temp_path, manifest_file)

        except Exception as e:
            os.remove(temp_path)
            raise
        finally:
            browser.close()

    print("\nProcessing complete.")
