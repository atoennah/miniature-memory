from .normalize import TextNormalizer
from .quality_filter import QualityFilter

class CleaningPipeline:
    """
    Orchestrates the cleaning process by combining normalization and quality filtering.
    """

    def __init__(self):
        self.normalizer = TextNormalizer()
        self.quality_filter = QualityFilter()

    def clean(self, text: str) -> str:
        """
        Applies the full cleaning pipeline to the input text.

        Args:
            text: The raw input text.

        Returns:
            The cleaned text, or an empty string if it fails quality checks.
        """
        # 1. Filter out blacklisted paragraphs (e.g., ads)
        text = self.quality_filter.filter_blacklisted_paragraphs(text)

        # 2. Normalize text (character filtering, whitespace)
        text = self.normalizer.normalize(text)

        # 3. Final validation
        is_valid, _ = self.quality_filter.validate_content(text)

        return text if is_valid else ""
