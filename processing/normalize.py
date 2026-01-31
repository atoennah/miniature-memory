import re

class TextNormalizer:
    """
    Handles standard text normalization tasks such as character filtering and
    whitespace normalization.
    """

    # Whitelist of characters: allows letters, numbers, basic punctuation, and whitespace.
    ALLOWED_CHARS = re.compile(r'[^a-zA-Z0-9\s.,?!\'"()-]')
    REPEATED_WHITESPACE = re.compile(r'[ \t]+')
    REPEATED_NEWLINES = re.compile(r'\n{3,}')

    def __init__(self):
        pass

    def filter_characters(self, text: str) -> str:
        """
        Removes characters not present in the ALLOWED_CHARS whitelist.

        Args:
            text: The raw text to clean.

        Returns:
            The text with non-allowed characters removed.
        """
        return self.ALLOWED_CHARS.sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalizes spaces and newlines. Replaces multiple spaces/tabs with
        a single space and reduces excessive newlines.

        Args:
            text: The text to normalize.

        Returns:
            The normalized text.
        """
        # Replace tabs and multiple spaces with a single space
        text = self.REPEATED_WHITESPACE.sub(' ', text)
        # Reduce sequences of 3 or more newlines to 2
        text = self.REPEATED_NEWLINES.sub('\n\n', text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """
        Applies a full suite of normalization operations to the text.

        Args:
            text: The raw text to normalize.

        Returns:
            The fully normalized text.
        """
        text = self.filter_characters(text)
        text = self.normalize_whitespace(text)
        return text
