import json
import os
import tempfile
from typing import Iterator, Union, Dict, List
from playwright.sync_api import sync_playwright, Page
from scraper.process import extract_text
from scraper.storage import save_raw_text
from scraper.crawler import find_story_urls_heuristically


class ManifestManager:
    """
    Handles operations on the JSONL manifest file, ensuring atomic updates.
    """

    def __init__(self, manifest_path: str):
        """
        Initializes the ManifestManager.

        Args:
            manifest_path: The path to the JSONL manifest file.
        """
        self.manifest_path = manifest_path
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(
                f"Manifest file not found: {self.manifest_path}"
            )

    def get_entries(self) -> Iterator[Union[Dict, str]]:
        """
        Yields entries from the manifest.

        Yields:
            A dictionary for valid JSON lines, or the raw string for
            invalid lines.
        """
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield line

    def save_manifest(self, entries: List[Union[Dict, str]]) -> None:
        """
        Writes entries back to the manifest file atomically.

        Args:
            entries: A list of entry dictionaries or raw strings to save.
        """
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.manifest_path)
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
                for entry in entries:
                    if isinstance(entry, dict):
                        temp_f.write(json.dumps(entry) + "\n")
                    else:
                        temp_f.write(entry + "\n")
            os.replace(temp_path, self.manifest_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


class StoryProcessor:
    """
    Handles the extraction and saving of a single story URL.
    """

    def __init__(self, page: Page):
        """
        Initializes the StoryProcessor.

        Args:
            page: The Playwright Page object to use for navigation.
        """
        self.page = page

    def process_story(self, url: str) -> bool:
        """
        Navigates to a story URL, extracts text, and saves it.

        Args:
            url: The URL of the story to process.

        Returns:
            True if processing was successful, False otherwise.
        """
        try:
            print(f"  -> Processing story: {url}")
            self.page.goto(url, timeout=60000)
            html = self.page.content()
            text = extract_text(html)
            if not text:
                print(f"    -> Warning: No text extracted from {url}")
                return False
            save_path = save_raw_text(url, text)
            return save_path is not None
        except Exception as e:
            print(f"    -> Failed to process {url}: {e}")
            return False


class CrawlerOrchestrator:
    """
    Orchestrates the browser lifecycle and the crawling process.
    """

    def __init__(self, manifest_manager: ManifestManager, debug: bool = False):
        """
        Initializes the CrawlerOrchestrator.

        Args:
            manifest_manager: The ManifestManager instance.
            debug: Whether to save debug snapshots.
        """
        self.manifest_manager = manifest_manager
        self.debug = debug

    def _save_debug_snapshot(self, page: Page) -> None:
        """
        Saves a screenshot and DOM dump for debugging.

        Args:
            page: The Playwright Page object to capture.
        """
        page.screenshot(path="debug_view.png")
        with open("debug_dom.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        print("  -> Saved debug snapshot and DOM dump.")

    def run(self) -> None:
        """
        Runs the crawling orchestration.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            story_processor = StoryProcessor(page)

            updated_entries = []
            for entry in self.manifest_manager.get_entries():
                # Guard Clause: Skip entries that are not new or are invalid
                if not isinstance(entry, dict) or entry.get("status") != "new":
                    updated_entries.append(entry)
                    continue

                index_url = entry.get('url')
                if not index_url:
                    entry["status"] = "rejected"
                    updated_entries.append(entry)
                    continue

                try:
                    print(f"Processing index URL: {index_url}")
                    page.goto(index_url, timeout=60000)
                    page.wait_for_timeout(5000)  # Wait for JS to settle

                    if self.debug:
                        self._save_debug_snapshot(page)

                    story_urls = find_story_urls_heuristically(page, index_url)

                    if not story_urls:
                        entry["status"] = "rejected"
                    else:
                        for story_url in story_urls:
                            story_processor.process_story(story_url)
                        entry["status"] = "crawled"

                except Exception as e:
                    print(f"  -> Error processing index {index_url}: {e}")

                updated_entries.append(entry)

            self.manifest_manager.save_manifest(updated_entries)
            browser.close()


def run_process(args):
    """
    Runs the heuristic crawl, fetch, extract, and save pipeline.
    This function treats URLs from the manifest as index pages, heuristically
    finds potential story links, and then processes them.

    Args:
        args: Command-line arguments.
    """
    manifest_path = getattr(args, 'manifest', 'dataset/metadata/urls.jsonl')
    debug = getattr(args, 'debug', False)

    try:
        manifest_manager = ManifestManager(manifest_path)
        orchestrator = CrawlerOrchestrator(manifest_manager, debug=debug)
        orchestrator.run()
        print("\nProcessing complete.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
