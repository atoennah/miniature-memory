
from typing import List

class QualityFilter:
    """
    Filters out content that doesn't meet quality standards (e.g., ads, noise).
    """
    def __init__(self, blacklist_keywords: List[str] = None):
        self.blacklist_keywords = blacklist_keywords or []

    def filter_paragraphs(self, text: str) -> str:
        """
        Splits text into paragraphs and removes those containing blacklisted keywords.
        """
        if not self.blacklist_keywords:
            return text

        paragraphs = text.split('\n')
        cleaned_paragraphs = []
        for p in paragraphs:
            if not any(keyword.lower() in p.lower() for keyword in self.blacklist_keywords):
                cleaned_paragraphs.append(p)

        return "\n".join(cleaned_paragraphs)
