import os
import re
import hashlib
from urllib.parse import urlparse
from datetime import datetime

# Define a constant for the maximum length of a filename component
MAX_FILENAME_COMPONENT_LENGTH = 50

def _sanitize_component(component: str) -> str:
    """Sanitizes a string to be used in a filename."""
    # Remove non-alphanumeric characters
    component = re.sub(r'[^a-zA-Z0-9_-]', '', component)
    # Truncate to a reasonable length to avoid overly long filenames
    return component[:MAX_FILENAME_COMPONENT_LENGTH].lower()

def save_raw_text(url: str, text_content: str) -> str | None:
    """
    Saves the raw extracted text to a file with a canonical name.

    Args:
        url (str): The original URL of the content.
        text_content (str): The raw text extracted from the URL.

    Returns:
        str | None: The path to the saved file, or None if saving fails.
    """
    if not text_content.strip():
        # Do not save empty files
        return None

    try:
        # 1. Generate UTC timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # 2. Derive Source Identifier (SRC) from the domain
        parsed_url = urlparse(url)
        # Get the domain and remove 'www.' if present
        domain = parsed_url.netloc.replace('www.', '')
        # Take the first part of the domain (e.g., 'blogspot' from 'myblog.blogspot.com')
        src = domain.split('.')[0]
        src = _sanitize_component(src)

        # 3. Derive Content Identifier (CID) from the URL path
        path_parts = [part for part in parsed_url.path.split('/') if part]
        cid = _sanitize_component(path_parts[-1]) if path_parts else 'index'

        # 4. Calculate Content Hash
        content_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()[:12]

        # 5. Construct the filename
        filename = f"{timestamp}__{src}__{cid}__{content_hash}.txt"

        # 6. Create the source-specific directory
        source_dir = os.path.join("dataset", "raw", f"source_{src}")
        os.makedirs(source_dir, exist_ok=True)

        # 7. Save the file
        filepath = os.path.join(source_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)

        return filepath
    except Exception as e:
        print(f"Error saving raw text for {url}: {e}")
        return None
