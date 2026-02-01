# [INJECTOR: THE STEALTH STRATEGY OF DYNAMIC FETCHING]
#
# Static HTML fetching (via `requests` or `curl`) is increasingly obsolete.
# Modern web servers use sophisticated Anti-Bot measures and client-side
# rendering (React/Vue/Next.js).
#
# Why Playwright?
# 1.  JavaScript Execution: We need a full browser engine to execute the scripts
#     that actually render the text.
# 2.  Behavioral Mimicry: By using a real browser, we produce a more "human"
#     fingerprint (TLS JA3, HTTP/2 frames, window properties).
# 3.  Event Synchronization: `wait_for_load_state('networkidle')` ensures we
#     capture the page only after all async content has arrived.
#
# Trafilatura: The "Signal from Noise" Filter
# Extracting text from a raw DOM is notoriously difficult. Trafilatura uses
# structural and linguistic analysis to identify the "Main Content" and strip
# away ads, sidebars, and trackers. It is our primary defense against "Data Pollution."

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import trafilatura

def fetch_html(url: str) -> str | None:
    """
    Fetches the HTML content for a given URL using a headless browser.

    Args:
        url (str): The URL to fetch.

    Returns:
        str | None: The HTML content as a string, or None if an error occurs.
    """

    # TODO [BOT-STEALTH]: To prevent IP-based blocking, we must integrate a
    # residential proxy rotator and randomize the User-Agent / Viewport.
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            # Set a longer timeout (60 seconds) for pages that are slow to load
            page.goto(url, timeout=60000)
            # Wait for the network to be idle, indicating the page has likely loaded
            page.wait_for_load_state('networkidle')
            html = page.content()
            browser.close()
            return html
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
