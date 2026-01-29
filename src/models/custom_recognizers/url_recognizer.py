import re
from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
from src.utils.constants import LANG_CODE


class CustUrlRecognizer(PatternRecognizer):
    """
    Recognize URLs using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex for detecting invalid URL patterns
    _invalid_url_chars_regex = re.compile(r'(?<!/)\.{2,}')

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE ,
            supported_entity: str = "URL",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
            name=f"{self.__class__.__name__}"
        )

    def invalidate_result(self, pattern_text: str) -> Optional[bool]:
        """
        Check if the pattern text cannot be validated as a URL.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalid
        """
        # Use precompiled regex for invalid URL checks
        return self._invalid_url_chars_regex.search(pattern_text) is not None
