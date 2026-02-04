# [INJECTOR: THE DISCOVERY PROTOCOL]
#
# Before we can crawl and extract content, we must find where it lives.
# This module acts as the "Scout" of the pipeline, utilizing DuckDuckGo
# Search (DDGS) to find high-probability content indices.
#
# WHY DUCKDUCKGO?
# 1. PRIVACY & LESS AGGRESSIVE BLOCKING: Unlike Google, DDG is often more
#    lenient towards automated discovery as long as it respects rate limits.
# 2. NO API KEYS REQUIRED: The `ddgs` library provides a clean interface
#    without the overhead of managing developer tokens or quotas.
#
# The goal here is to seed the `urls.jsonl` manifest with "Index URLs"
# (e.g., a blog's home page or a search results page) which the heuristic
# crawler will later expand.

from ddgs import DDGS
import datetime

def discover_urls(query, num_results=10):
    """
    Discovers URLs for a given query using DuckDuckGo Search.

    Args:
        query (str): The search query.
        num_results (int): The number of results to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a discovered URL.
    """
    urls = []
    try:
        # [INJECTOR NOTE]: DDGS is a context manager. It handles the session
        # lifecycle, including connection pooling.
        with DDGS() as ddgs:
            # [INJECTOR NOTE]: We use `ddgs.text` for standard web results.
            # `max_results` should be kept reasonable (e.g., < 100) to
            # avoid triggering anti-bot protection.
            results = ddgs.text(query, max_results=num_results)
            for r in results:
                urls.append({
                    "url": r['href'],
                    "query": query,
                    "discovered_at": datetime.datetime.utcnow().isoformat(),
                    # [INJECTOR NOTE]: Every discovered URL starts as "new".
                    # The `process` command will transition this to "crawled"
                    # or "rejected".
                    "status": "new"
                })
    except Exception as e:
        print(f"An error occurred during search: {e}")
    return urls
