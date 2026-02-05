"""
Core processing utilities for fetching and extracting text from web pages.
"""
from playwright.sync_api import sync_playwright
import trafilatura
from typing import Optional


def fetch_html(url: str) -> Optional[str]:
    """
    Fetches the HTML content for a given URL using a headless browser.

    This function launches a temporary Chromium instance to load the page,
    waiting for the network to be idle to ensure dynamic content is loaded.

    Args:
        url: The URL to fetch.

    Returns:
        The HTML content of the page as a string, or None if the fetch fails.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Set a longer timeout (60 seconds) for slow pages
            page.goto(url, timeout=60000)
            # Wait for the network to be idle
            page.wait_for_load_state('networkidle')
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"Error fetching {url} with Playwright: {e}")
        return None


def extract_text(html: Optional[str]) -> str:
    """
    Extracts the main story text from HTML using trafilatura.

    Trafilatura is used for its superior ability to identify the primary
    content area while stripping boilerplate like navigation, ads, and footers.

    Args:
        html: The raw HTML content of the page.

    Returns:
        The extracted plain text content. Returns an empty string if HTML is
        None or if extraction fails.
    """
    if not html:
        return ""

    # Trafilatura is a library specifically designed to extract the main
    # text content from a webpage, filtering out menus, ads, and footers.
    extracted = trafilatura.extract(html)
    return extracted if extracted else ""
