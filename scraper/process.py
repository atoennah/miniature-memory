import requests
from bs4 import BeautifulSoup

def fetch_html(url: str) -> str | None:
    """
    Fetches the HTML content for a given URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        str | None: The HTML content as a string, or None if the request fails.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_text(html: str) -> str:
    """
    Extracts readable story text from HTML, focusing on paragraph tags.

    Args:
        html (str): The HTML content of the page.

    Returns:
        str: The extracted plain text content.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, 'html.parser')

    # A simple but effective strategy for story text is to concatenate the text
    # from all paragraph (<p>) tags. This tends to filter out most UI noise.
    paragraphs = soup.find_all('p')

    # Join the text from all paragraphs with a double newline for separation.
    story_text = "\n\n".join(p.get_text().strip() for p in paragraphs)

    return story_text
