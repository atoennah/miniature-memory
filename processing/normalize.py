
import re

class TextNormalizer:
    """
    Standard text normalization for story data.
    """
    def __init__(self):
        self.repeated_whitespace = re.compile(r'[ \t]+')
        self.repeated_newlines = re.compile(r'\n{3,}')
        # Whitelist: letters, numbers, basic punctuation, and whitespace.
        self.allowed_chars = re.compile(r'[^a-zA-Z0-9\s.,?!\'"()-]')

    def normalize(self, text: str) -> str:
        """
        Applies a sequence of normalization steps to the input text.
        """
        if not text:
            return ""

        # Remove non-allowed characters
        text = self.allowed_chars.sub('', text)

        # Normalize whitespace
        text = self.repeated_whitespace.sub(' ', text)

        # Normalize newlines
        text = self.repeated_newlines.sub('\n\n', text)

        return text.strip()
