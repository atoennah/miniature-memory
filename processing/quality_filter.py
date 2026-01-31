from typing import List, Set

class QualityFilter:
    """
    Provides methods for assessing text quality, including keyword filtering,
    language detection, and printable character analysis.
    """

    # --- Filtering Constants ---
    GAMBLING_AD_KEYWORDS = [
        "slot gacor", "judi online", "daftar segera", "bonus new member",
        "zeus", "pragmatic play", "agen bola", "togel"
    ]

    INDONESIAN_STOP_WORDS = {
        "yang", "dan", "aku", "dia", "dengan", "tidak", "ini", "itu", "ke", "di"
    }

    def __init__(
        self,
        min_file_length: int = 50,
        min_printable_ratio: float = 0.85,
        min_stop_word_count: int = 5
    ):
        self.min_file_length = min_file_length
        self.min_printable_ratio = min_printable_ratio
        self.min_stop_word_count = min_stop_word_count

    def filter_blacklisted_paragraphs(self, text: str) -> str:
        """
        Splits text into paragraphs and removes any that contain blacklisted keywords.

        Args:
            text: The text to filter.

        Returns:
            The filtered text with blacklisted paragraphs removed.
        """
        paragraphs = text.split('\n')
        cleaned_paragraphs = []
        for p in paragraphs:
            if not any(keyword in p.lower() for keyword in self.GAMBLING_AD_KEYWORDS):
                cleaned_paragraphs.append(p)
        return "\n".join(cleaned_paragraphs)

    def is_indonesian(self, text: str) -> bool:
        """
        Checks if the text is likely Indonesian based on common stop-word count.

        Args:
            text: The text to check.

        Returns:
            True if it meets the Indonesian stop-word threshold.
        """
        words = set(text.lower().split())
        stop_word_matches = words.intersection(self.INDONESIAN_STOP_WORDS)
        return len(stop_word_matches) >= self.min_stop_word_count

    def get_printable_ratio(self, text: str) -> float:
        """
        Calculates the ratio of printable characters in the text.

        Args:
            text: The text to analyze.

        Returns:
            The ratio of printable characters (0.0 to 1.0).
        """
        if not text:
            return 0.0

        def is_printable(char):
            return char.isprintable() or char in ('\n', '\r', '\t')

        printable_chars = sum(1 for char in text if is_printable(char))
        return printable_chars / len(text)

    def validate_content(self, text: str) -> tuple[bool, str]:
        """
        Runs multiple quality checks on the text.

        Args:
            text: The text to validate.

        Returns:
            A tuple of (is_valid, reason).
        """
        if len(text) < self.min_file_length:
            return False, f"Too short ({len(text)} chars)"

        ratio = self.get_printable_ratio(text)
        if ratio < self.min_printable_ratio:
            return False, f"Low printable ratio ({ratio:.2f})"

        if not self.is_indonesian(text):
            return False, "Failed language check (not enough Indonesian stop-words)"

        return True, "Valid"
