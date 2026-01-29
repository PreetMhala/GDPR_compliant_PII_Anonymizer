from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
import re
from src.utils.constants import LANG_CODE


class CustIbanRecognizer(PatternRecognizer):
    """
    Recognize IBAN Numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex for performance
    _non_digit_regex = re.compile(r"\D")
    _invalid_chars_regex = re.compile(r'[^A-Z0-9]')
    _illegal_whitespace_regex = re.compile(r'[\n\t]')

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE        ,
            supported_entity: str = "IBAN",
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
        Check if the pattern text cannot be validated as an IBAN.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        checks = [
            15 > len(self._non_digit_regex.sub("", pattern_text)),
            self._invalid_chars_regex.search(pattern_text) is not None,
            self._illegal_whitespace_regex.search(pattern_text) is not None,
            ]
        return any(checks)
