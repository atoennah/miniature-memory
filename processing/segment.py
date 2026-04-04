
class Segmenter:
    """
    Handles structural markers and segmentation of stories.
    """
    def __init__(self, start_marker="<|story_start|>", end_marker="<|end_of_text|>"):
        self.start_marker = start_marker
        self.end_marker = end_marker

    def wrap(self, text: str) -> str:
        return f"{self.start_marker}\n{text}\n{self.end_marker}"
