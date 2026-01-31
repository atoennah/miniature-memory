from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import trafilatura
from typing import Optional

class BrowserManager:
    """
    Manages a persistent Playwright browser instance to avoid the overhead
    of launching a new browser for every request.
    """
    _instance = None

    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def fetch(self, url: str) -> str:
        # Each fetch uses a new page (tab), which is much faster than a new browser.
        page = self.browser.new_page()
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state('networkidle')
            return page.content()
        finally:
            page.close()

    @classmethod
    def close_instance(cls):
        if cls._instance:
            cls._instance.browser.close()
            cls._instance.playwright.stop()
            cls._instance = None

def fetch_html(url: str) -> Optional[str]:
    """
    Fetches the HTML content for a given URL using a persistent headless browser.

    Args:
        url (str): The URL to fetch.

    Returns:
        str | None: The HTML content as a string, or None if an error occurs.
    """
    try:
        return BrowserManager.get_instance().fetch(url)
    except Exception as e:
        print(f"Error fetching {url} with Playwright: {e}")
        return None

def extract_text(html: str) -> str:
    """
    Extracts the main story text from HTML using trafilatura.
    This is much more effective at removing boilerplate than a simple
    paragraph-tag search.

    Args:
        html (str): The HTML content of the page.

    Returns:
        str: The extracted plain text content.
    """
    if not html:
        return ""

    # Trafilatura is a library specifically designed to extract the main
    # text content from a webpage, filtering out menus, ads, and footers.
    return trafilatura.extract(html)
