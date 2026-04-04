
from typing import List, Set

class QualityFilter:
    """
    Filters text based on language quality and presence of prohibited content.
    """
    INDONESIAN_STOP_WORDS: Set[str] = {
        "yang", "dan", "aku", "dia", "dengan", "tidak", "ini", "itu", "ke", "di",
        "ada", "dari", "saya", "untuk", "pada", "sudah", "bisa", "akan"
    }

    def __init__(self, blacklist_keywords: List[str] = None, min_stop_word_count: int = 5):
        self.blacklist_keywords = blacklist_keywords or []
        self.min_stop_word_count = min_stop_word_count

    def is_indonesian(self, text: str) -> bool:
        """
        Heuristic to check if text is Indonesian based on common stop words.
        """
        words = set(text.lower().split())
        matches = words.intersection(self.INDONESIAN_STOP_WORDS)
        return len(matches) >= self.min_stop_word_count

    def contains_noise(self, text: str) -> bool:
        """
        Checks if the text contains blacklisted keywords or promotional boilerplate.
        """
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.blacklist_keywords)

    def filter_paragraphs(self, text: str) -> str:
        """
        Splits text into paragraphs and removes those that contain noise.
        """
        paragraphs = text.split('\n')
        cleaned_paragraphs = [p for p in paragraphs if not self.contains_noise(p)]
        return '\n'.join(cleaned_paragraphs).strip()

    def validate(self, text: str) -> bool:
        """
        High-level validation: must be Indonesian and not empty after noise filtering.
        """
        if not text:
            return False

        # Check language
        if not self.is_indonesian(text):
            return False

        return True
