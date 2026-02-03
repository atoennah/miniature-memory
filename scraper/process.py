import atexit
from playwright.sync_api import sync_playwright, Browser, Page, Playwright
import trafilatura

class BrowserManager:
    """
    Manages the lifecycle of a Playwright browser instance.

    This class implements the "Persistent Browser" pattern, ensuring that a
    single browser instance is reused across multiple requests. This
    significantly reduces the overhead of launching and closing the browser
    process for every URL, which is the primary bottleneck in web scraping.
    """

    def __init__(self, headless: bool = True):
        """
        Initializes the BrowserManager.

        Args:
            headless (bool): Whether to run the browser in headless mode.
                             Defaults to True for production scraping.
        """
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    def start(self) -> Browser:
        """
        Starts the Playwright instance and launches the browser if not already running.

        Returns:
            Browser: The active Playwright Browser instance.
        """
        if not self._browser:
            print("⚡ Bolt: Initializing persistent browser instance...")
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
        return self._browser

    def shutdown(self):
        """
        Gracefully shuts down the browser and the Playwright instance.
        """
        if self._browser:
            print("⚡ Bolt: Shutting down persistent browser...")
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def get_page(self) -> Page:
        """
        Creates and returns a new page in the persistent browser.

        Returns:
            Page: A new Playwright Page object.
        """
        browser = self.start()
        return browser.new_page()

    def fetch_page_content(self, url: str, timeout: int = 60000) -> str | None:
        """
        Fetches the full HTML content of a URL.

        This method encapsulates the navigation logic, including waiting for the
        'networkidle' state to ensure that dynamic JavaScript content has
        finished loading.

        Args:
            url (str): The target URL to fetch.
            timeout (int): The maximum time to wait for the page to load (ms).

        Returns:
            str | None: The HTML content as a string, or None if navigation fails.
        """
        page = self.get_page()
        try:
            page.goto(url, timeout=timeout)
            # Wait for the network to be idle, indicating the page has likely loaded
            page.wait_for_load_state('networkidle')
            return page.content()
        except Exception as e:
            print(f"Error fetching {url} with BrowserManager: {e}")
            return None
        finally:
            page.close()

# Global instance for easy reuse across the module
_browser_manager = BrowserManager()

# Ensure the browser is shut down when the script exits
atexit.register(_browser_manager.shutdown)

def fetch_html(url: str) -> str | None:
    """
    Legacy wrapper for fetching HTML content.
    Now uses the persistent BrowserManager.
    """
    return _browser_manager.fetch_page_content(url)

def extract_text(html: str) -> str:
    """
    Extracts the main story text from HTML using trafilatura.

    Trafilatura is a high-performance library specifically designed to
    identify and extract the core narrative content from a webpage while
    filtering out UI noise like menus, ads, and footers.

    Args:
        html (str): The HTML content of the page.

    Returns:
        str: The extracted plain text content.
    """
    if not html:
        return ""

    # trafilatura.extract is deterministic and optimized for high-quality extraction.
    extracted = trafilatura.extract(html)
    return extracted if extracted else ""
