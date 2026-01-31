# [INJECTOR: THE ARCHITECTURE OF STEALTH CRAWLING]
#
# Web scraping in the modern era is no longer a simple matter of fetching HTML.
# Sites increasingly use sophisticated bot-detection mechanisms, including:
# 1.  User-Agent Analysis: Checking if the browser identifies as a known bot.
# 2.  JavaScript Execution: Verifying that the client can execute complex JS, which
#     simple `requests` or `urllib` calls cannot do.
# 3.  TLS Fingerprinting: Analyzing the handshake to see if it matches a real browser.
# 4.  Behavioral Analysis: Monitoring the timing and sequence of requests.
#
# To counter this, we use Playwright, a powerful browser automation library. It
# launches a real, headless Chromium instance, allowing us to:
# -   Execute all JavaScript, ensuring lazy-loaded content is captured.
# -   Mimic real browser headers and behavior.
# -   Handle `networkidle` states to wait for asynchronous content to load.
#
# This module serves as the primary gateway for high-fidelity content extraction,
# transforming a hostile web into a structured dataset for LLM training.
#
# References:
# - Playwright Documentation: https://playwright.dev/python/docs/intro
# - Trafilatura Documentation: https://trafilatura.readthedocs.io/
# - TLS Fingerprinting Explained: https://browserleaks.com/tls

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
    # [INJECTOR: CONTENT VS BOILERPLATE]
    #
    # A major challenge in web scraping is "Signal-to-Noise" ratio. A typical
    # webpage consists of 80% boilerplate (menus, ads, sidebars, footers) and
    # only 20% actual content.
    #
    # We use `trafilatura`, which employs sophisticated heuristics based on
    # HTML structure, text density, and tag analysis to identify the "main"
    # content block. This is far more robust than manually selecting CSS
    # selectors, which vary wildly between sites.

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
