from playwright.sync_api import Page
from urllib.parse import urljoin
from typing import List, Set

def find_story_urls_heuristically(page: Page, index_url: str) -> List[str]:
    """
    Finds story URLs on a page using a heuristic-based approach.

    This function scrolls the page to trigger lazy-loaded content, then
    analyzes all links based on their URL structure and text content
    to identify links that are likely to be stories.

    Args:
        page: The Playwright Page object to analyze.
        index_url: The URL of the page being analyzed (for resolving relative links).

    Returns:
        A list of unique, absolute URLs that are likely to be stories.
    """
    print(f"Heuristically analyzing links on: {index_url}")

    # Scroll to the bottom of the page to trigger any lazy-loading scripts
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(3000)  # Wait for content to potentially load

    links = page.locator("a").all()
    valid_urls: Set[str] = set()

    for link in links:
        url = link.get_attribute("href")
        text = link.inner_text().strip()

        # --- Heuristic 1: URL Structure ---
        if not url:
            continue
        # Reject short URLs, mailto links, or javascript links
        if len(url) < 10 or url.startswith("mailto:") or url.startswith("javascript:"):
            continue
        # Reject common non-story paths
        non_story_patterns = ["/tag/", "/category/", "/search/", "/author/", "/archive/", "/page/", "login", "signup", "faq"]
        if any(pattern in url for pattern in non_story_patterns):
            continue

        # --- Heuristic 2: Text Density ---
        word_count = len(text.split())
        # Story titles are usually between 3 and 25 words.
        if not (3 <= word_count <= 25):
            continue

        # --- Heuristic 3: Reject Common Non-Story Link Text ---
        non_story_text = ["home", "about", "contact", "privacy", "terms", "shop", "forum"]
        if text.lower() in non_story_text:
            continue

        # If it passes all heuristics, it's likely a story link.
        absolute_url = urljoin(index_url, url)
        valid_urls.add(absolute_url)

    print(f"Found {len(valid_urls)} potential story URLs.")
    return list(valid_urls)
