# [INJECTOR: THE STEALTH STRATEGY & CONTENT DISTILLATION]
#
# Scraping modern web pages is no longer a matter of simple `requests.get()`.
# Pages are dynamic, protected by anti-bot measures, and cluttered with
# non-content boilerplate.
#
# This module employs a dual-pronged strategy:
# 1.  STEALTH BROWSER EMULATION: We use Playwright to launch a headless Chromium
#     instance. This allows us to execute JavaScript, handle cookies, and
#     bypass basic fingerprinting. By waiting for `networkidle`, we ensure
#     that asynchronous content (like the actual story text) is fully rendered.
#
# 2.  CONTENT DISTILLATION: Instead of brittle CSS selectors, we use Trafilatura.
#     Trafilatura uses a combination of algorithms (density-based, pattern
#     recognition) to strip away the "noise" (ads, footers, sidebars) and
#     extract the core "signal" (the story).

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
            # [INJECTOR NOTE]: 60s timeout accounts for high-latency sites or
            # slow JS execution.
            page.goto(url, timeout=60000)
            # [INJECTOR NOTE]: 'networkidle' is the gold standard for dynamic
            # pages. It waits until there are no more than 0 network
            # connections for at least 500ms.
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

    # [INJECTOR NOTE]: Trafilatura is preferred here because it maintains
    # better structural integrity of the text (e.g., preserving paragraphs)
    # compared to raw BeautifulSoup text extraction, which often results
    # in "collapsed" text blocks.
    return trafilatura.extract(html)
