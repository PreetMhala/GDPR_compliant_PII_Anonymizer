from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
import re
from src.utils.constants import LANG_CODE


class CustIanRecognizer(PatternRecognizer):
    """
    Recognize IAN patterns using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex for performance
    _compiled_identifier_regex = re.compile(r"\d{4,}@[\w.-]+")
    _compiled_illegal_chars = re.compile(r'[\n\t]')

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE        ,
            supported_entity: str = "INTERNET_ACCESS_NR",
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

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Check if the pattern text cannot be validated as an IAN.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        try:
            parts = pattern_text.split('#')
            if len(parts) != 2:
                return True

            number_part, identifier_part = parts

            if len(number_part) != 21 or not number_part.isdigit():
                return True

            if not self._compiled_identifier_regex.match(identifier_part):
                return True

            if self._compiled_illegal_chars.search(pattern_text):
                return True

        except Exception:
            return True

        return False
