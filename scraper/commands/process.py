# [INJECTOR: THE ARCHITECTURE OF A ROBUST SCRAPER ORCHESTRATOR]
#
# Scraping is inherently fragile. Websites change, networks fail, and data can be
# malformed. This orchestrator is designed with three core pillars of robustness:
#
# 1.  Atomic State Management: The manifest (urls.jsonl) is our "Source of Truth".
#     To prevent corruption during crashes, we never edit the file in-place. We use
#     a 'ManifestManager' to handle atomic swaps via temporary files.
#
# 2.  Separation of Concerns: The logic is decomposed into specialized classes.
#     - `ManifestManager`: Handles the persistence layer and manifest iteration.
#     - `StoryProcessor`: Handles the browser-based interaction and extraction.
#     - `run_process`: Acts as the high-level orchestrator.
#
# 3.  Stealth & Heuristics: We use Playwright for full browser emulation to bypass
#     basic bot detection and use heuristic discovery to find stories without
#     relying on brittle, hardcoded CSS selectors.
#
# This modular approach makes the pipeline testable, maintainable, and resilient
# to the chaos of the open web.

import json
import os
import tempfile
from typing import List, Dict, Any, Optional, Iterator
from playwright.sync_api import sync_playwright, Page
from scraper.process import extract_text
from scraper.storage import save_raw_text
from scraper.crawler import find_story_urls_heuristically

class ManifestManager:
    """
    Handles atomic operations and iteration for the URL manifest file.

    Attributes:
        filepath: The path to the JSONL manifest file.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def exists(self) -> bool:
        """Checks if the manifest file exists."""
        return os.path.exists(self.filepath)

    def entries(self) -> Iterator[Dict[str, Any] | str]:
        """
        Iterates over entries in the manifest.

        Yields:
            A dictionary representing a single URL entry, or the raw string
            if the line is not valid JSON (to preserve malformed lines).
        """
        if not self.exists():
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield line

class StoryProcessor:
    """
    Encapsulates the logic for crawling index pages and extracting story content.

    Attributes:
        page: A Playwright Page instance used for navigation.
    """
    def __init__(self, page: Page):
        self.page = page

    def process_index(self, index_url: str) -> List[str]:
        """
        Navigates to an index page and discovers story URLs using heuristics.

        Args:
            index_url: The URL of the index page to process.

        Returns:
            A list of discovered story URLs.
        """
        print(f"Processing index URL: {index_url}")
        try:
            # Navigate with a generous timeout for slow adult-content sites
            self.page.goto(index_url, timeout=60000)
            self.page.wait_for_timeout(5000)  # Allow JS and dynamic content to settle

            # Save debug artifacts for troubleshooting extraction failures
            self.page.screenshot(path="debug_view.png")
            with open("debug_dom.html", "w", encoding="utf-8") as f:
                f.write(self.page.content())

            return find_story_urls_heuristically(self.page, index_url)
        except Exception as e:
            print(f"  [!] Failed to process index {index_url}: {e}")
            return []

    def process_story(self, story_url: str) -> bool:
        """
        Navigates to a story page, extracts the main text content, and saves it.

        Args:
            story_url: The URL of the story to process.

        Returns:
            True if the story was successfully processed and saved, False otherwise.
        """
        print(f"  -> Processing story: {story_url}")
        try:
            self.page.goto(story_url, timeout=60000)
            html = self.page.content()
            text = extract_text(html)

            if not text or not text.strip():
                print(f"    [!] No text extracted from {story_url}")
                return False

            save_raw_text(story_url, text)
            return True
        except Exception as e:
            print(f"    [!] Failed to process story {story_url}: {e}")
            return False

def run_process(args: Any) -> None:
    """
    The main entry point for the manifest processing command.

    Orchestrates the lifecycle of the browser and iterates through the manifest
    to process any 'new' URLs.

    Args:
        args: CLI arguments (currently unused but required by the dispatcher).
    """
    manifest_path = "dataset/metadata/urls.jsonl"
    manager = ManifestManager(manifest_path)

    if not manager.exists():
        print("URL manifest file not found. Run 'search' first.")
        return

    # Use a single Playwright context for the entire run to optimize performance
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        processor = StoryProcessor(page)

        # Prepare for atomic manifest update
        temp_dir = os.path.dirname(manifest_path)
        temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

        try:
            with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
                for entry in manager.entries():
                    # Only process URLs that are valid dicts and haven't been visited yet
                    if isinstance(entry, dict) and entry.get("status") == "new":
                        index_url = entry['url']
                        story_urls = processor.process_index(index_url)

                        if not story_urls:
                            entry["status"] = "rejected"
                        else:
                            for s_url in story_urls:
                                processor.process_story(s_url)
                            entry["status"] = "crawled"

                    # Persist the (possibly updated) entry to the temporary manifest
                    if isinstance(entry, dict):
                        temp_f.write(json.dumps(entry) + "\n")
                    else:
                        # entry is the raw malformed line, preserve it as-is
                        temp_f.write(entry)

            # Atomic swap to finalize the update
            os.replace(temp_path, manifest_path)
            print("\nProcessing complete.")

        except Exception as e:
            print(f"CRITICAL ERROR during manifest processing: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            browser.close()
