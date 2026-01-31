class Segmenter:
    """
    Handles the segmentation of text and the addition of structural control tokens.
    """

    STORY_START = "<|story_start|>"
    END_OF_TEXT = "<|end_of_text|>"

    def __init__(self):
        pass

    def add_story_markers(self, text: str) -> str:
        """
        Wraps the text with story start and end markers.

        Args:
            text: The narrative text.

        Returns:
            The text with markers added.
        """
        # Ensure we don't double-add markers if they are already there
        if text.strip().startswith(self.STORY_START) and text.strip().endswith(self.END_OF_TEXT):
            return text

        return f"{self.STORY_START}\n{text.strip()}\n{self.END_OF_TEXT}\n"
