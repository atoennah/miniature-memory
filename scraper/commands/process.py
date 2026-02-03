import json
import os
import tempfile
from scraper.process import extract_text, _browser_manager
from scraper.storage import save_raw_text
from scraper.crawler import find_story_urls_heuristically

class ManifestManager:
    """
    Handles atomic operations on the URL manifest file.

    Ensures that updates to the manifest are durable and consistent by using
    a temporary file and atomic replacement.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def walk_and_update(self, processor_callback):
        """
        Iterates through the manifest and applies a callback to each 'new' entry.

        Args:
            processor_callback (callable): A function that takes a URL data dict
                                           and returns the updated dict.
        """
        if not os.path.exists(self.filepath):
            print(f"⚡ Bolt: Manifest file not found at {self.filepath}. Run 'search' first.")
            return

        # Create a temporary file in the same directory for atomic replacement
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.filepath))

        try:
            with os.fdopen(temp_fd, 'w', encoding="utf-8") as temp_f:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            url_data = json.loads(line)
                            if url_data.get("status") == "new":
                                # Process the entry using the provided callback
                                url_data = processor_callback(url_data)

                            temp_f.write(json.dumps(url_data) + "\n")
                        except json.JSONDecodeError:
                            # Keep malformed lines as-is
                            temp_f.write(line)

            # Atomic replacement of the original manifest
            os.replace(temp_path, self.filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"⚡ Bolt: Critical error during manifest update: {e}")
            raise

class StoryProcessor:
    """
    Coordinates the crawling of index pages and extraction of individual stories.

    This class encapsulates the scraping workflow, separating the high-level
    process orchestration from the low-level browser operations.
    """

    def __init__(self, browser_manager, debug: bool = False):
        self.browser_manager = browser_manager
        self.debug = debug

    def process_index_entry(self, entry: dict) -> dict:
        """
        Processes a single index URL entry from the manifest.

        Heuristically discovers story links on the page and then processes
        each discovered story.
        """
        index_url = entry.get('url')
        if not index_url:
            return entry

        print(f"⚡ Bolt: Processing index URL: {index_url}")
        page = self.browser_manager.get_page()

        try:
            # Set a generous timeout for index pages which might be heavy
            page.goto(index_url, timeout=60000)
            # Give lazy-loading content some time to settle
            page.wait_for_timeout(5000)

            if self.debug:
                self._capture_debug_info(page, "index_debug")

            # Heuristic discovery of story links
            story_urls = find_story_urls_heuristically(page, index_url)

            if not story_urls:
                print(f"  -> No story URLs found. Marking as rejected.")
                entry["status"] = "rejected"
            else:
                print(f"  -> Found {len(story_urls)} potential stories.")
                for story_url in story_urls:
                    self.process_story_page(story_url)
                entry["status"] = "crawled"

        except Exception as e:
            print(f"  -> Error processing index {index_url}: {e}")
            # If it fails, we keep it as 'new' to retry later, or we could mark as 'error'
        finally:
            page.close()

        return entry

    def process_story_page(self, story_url: str):
        """
        Fetches a single story page, extracts its text, and saves it.
        """
        print(f"    -> Extracting: {story_url}")
        page = self.browser_manager.get_page()
        try:
            page.goto(story_url, timeout=60000)
            # Wait for content to stabilize
            page.wait_for_load_state('networkidle')

            html = page.content()
            text = extract_text(html)

            if text:
                saved_path = save_raw_text(story_url, text)
                if saved_path:
                    print(f"      [✅ SAVED] {os.path.basename(saved_path)}")
            else:
                print(f"      [⚠️ EMPTY] No text extracted from {story_url}")

        except Exception as e:
            print(f"    -> [❌ FAILED] {story_url}: {e}")
        finally:
            page.close()

    def _capture_debug_info(self, page, prefix: str):
        """Captures a screenshot and DOM dump for debugging purposes."""
        try:
            screenshot_path = f"{prefix}.png"
            dom_path = f"{prefix}.html"
            page.screenshot(path=screenshot_path)
            with open(dom_path, "w", encoding="utf-8") as f:
                f.write(page.content())
            print(f"  -> [DEBUG] Saved snapshot to {screenshot_path}")
        except Exception as e:
            print(f"  -> [DEBUG] Failed to capture debug info: {e}")

def run_process(args):
    """
    The main entry point for the 'process' command.
    Orchestrates the manifest update and story processing.
    """
    manifest_file = "dataset/metadata/urls.jsonl"

    # Initialize our modular components
    manager = ManifestManager(manifest_file)
    processor = StoryProcessor(_browser_manager, debug=args.debug)

    print("⚡ Bolt: Starting story extraction pipeline...")

    # Run the walk-and-update cycle
    manager.walk_and_update(processor.process_index_entry)

    print("\n⚡ Bolt: Processing complete.")
