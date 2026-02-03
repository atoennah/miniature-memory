"""
Heuristic-based story discovery for web scraping.

This module provides the logic to identify potential story links on a webpage
using probabilistic rules (heuristics) based on URL structure, link text,
and content density.
"""

from playwright.sync_api import Page
from urllib.parse import urljoin
from typing import List, Set

class StoryLinkHeuristics:
    """
    A collection of constants and rules used to identify story links.

    These heuristics are based on common patterns observed in Indonesian
    story aggregators and platforms like Wattpad.
    """

    # Minimum length of a URL to be considered a story (to filter out homepages/short nav)
    MIN_URL_LENGTH = 10

    # URL patterns that typically point to index pages or system pages, not stories.
    NON_STORY_URL_PATTERNS = [
        "/tag/", "/category/", "/search/", "/author/",
        "/archive/", "/page/", "login", "signup", "faq",
        "javascript:", "mailto:"
    ]

    # Typical word count range for a story title link.
    MIN_TITLE_WORDS = 3
    MAX_TITLE_WORDS = 25

    # Common navigation link text that should be ignored.
    NON_STORY_LINK_TEXT = {
        "home", "about", "contact", "privacy", "terms",
        "shop", "forum", "help", "next", "prev", "previous"
    }

def find_story_urls_heuristically(page: Page, index_url: str) -> List[str]:
    """
    Finds story URLs on a page using a heuristic-based approach.

    This function performs the following steps:
    1. Scrolls the page to trigger lazy-loaded content (common in modern SPAs).
    2. Collects all anchor tags on the page.
    3. Filters links based on URL length and excluded patterns.
    4. Filters links based on the word density of their inner text.
    5. Resolves relative links to absolute URLs.

    Args:
        page (Page): The Playwright Page object to analyze.
        index_url (str): The URL of the page being analyzed.

    Returns:
        List[str]: A list of unique, absolute URLs that are likely stories.
    """
    print(f"⚡ Bolt: Analyzing links on: {index_url}")

    # Step 1: Trigger Lazy Loading
    # We scroll to the bottom to ensure that any infinite-scroll or lazy-load
    # scripts have a chance to populate the DOM with more story links.
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(3000)

    links = page.locator("a").all()
    valid_urls: Set[str] = set()

    for link in links:
        try:
            url = link.get_attribute("href")
            text = link.inner_text().strip()

            if not url:
                continue

            # --- Heuristic 1: URL Filtering ---
            # Reject short URLs or those starting with non-http protocols.
            if len(url) < StoryLinkHeuristics.MIN_URL_LENGTH:
                continue

            if any(pattern in url.lower() for pattern in StoryLinkHeuristics.NON_STORY_URL_PATTERNS):
                continue

            # --- Heuristic 2: Link Text Analysis ---
            # Most story titles consist of a few words. Single words are usually nav links.
            words = text.split()
            word_count = len(words)
            if not (StoryLinkHeuristics.MIN_TITLE_WORDS <= word_count <= StoryLinkHeuristics.MAX_TITLE_WORDS):
                continue

            # Filter out common boilerplate navigation text.
            if text.lower() in StoryLinkHeuristics.NON_STORY_LINK_TEXT:
                continue

            # --- Heuristic 3: Resolution ---
            # Convert relative links (e.g., "/story/123") to absolute URLs.
            absolute_url = urljoin(index_url, url)

            # Basic sanity check: ensure we didn't resolve to the index page itself
            if absolute_url.rstrip('/') == index_url.rstrip('/'):
                continue

            valid_urls.add(absolute_url)
        except Exception:
            # If a link disappears from the DOM or has issues, skip it.
            continue

    print(f"  -> Discovered {len(valid_urls)} potential story links.")
    return list(valid_urls)
