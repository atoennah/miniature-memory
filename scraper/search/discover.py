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
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            for r in results:
                urls.append({
                    "url": r['href'],
                    "query": query,
                    "discovered_at": datetime.datetime.utcnow().isoformat(),
                    "status": "new"
                })
    except Exception as e:
        print(f"An error occurred during search: {e}")
    return urls
