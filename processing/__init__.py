
from .normalize import TextNormalizer
from .quality_filter import QualityFilter
from .segment import Segmenter

class CleaningPipeline:
    """
    Orchestrates the text cleaning process using multiple modular components.
    """
    def __init__(self, blacklist_keywords=None):
        self.normalizer = TextNormalizer()
        self.quality_filter = QualityFilter(blacklist_keywords=blacklist_keywords)
        self.segmenter = Segmenter()

    def process(self, text: str) -> str:
        # 1. Quality Filtering (per paragraph)
        text = self.quality_filter.filter_paragraphs(text)
        # 2. Normalization
        text = self.normalizer.normalize(text)
        # 3. Segmentation (Optional, usually done during data prep,
        # but included here for completeness of the modular vision)
        # text = self.segmenter.wrap(text)

        return text
