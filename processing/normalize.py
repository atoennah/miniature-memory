
import re

class TextNormalizer:
    """
    Handles basic text cleaning and whitespace normalization.
    """
    def __init__(self, allowed_chars_pattern=r'[^a-zA-Z0-9\s.,?!\'"()-]'):
        self.allowed_chars = re.compile(allowed_chars_pattern)
        self.repeated_whitespace = re.compile(r'[ \t]+')
        self.repeated_newlines = re.compile(r'\n{3,}')

    def normalize(self, text: str) -> str:
        # Remove non-allowed characters
        text = self.allowed_chars.sub('', text)
        # Normalize whitespace
        text = self.repeated_whitespace.sub(' ', text)
        # Reduce repeated newlines
        text = self.repeated_newlines.sub('\n\n', text)
        return text.strip()
