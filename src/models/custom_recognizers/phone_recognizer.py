from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
import re
from src.utils.constants import LANG_CODE


class CustPhoneNumberRecognizer(PatternRecognizer):
    """
    Recognize Phone Numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex patterns for performance
    _non_digit_regex = re.compile(r"\D")
    _invalid_char_regex = re.compile(r'[^+_.\d\s()-]')
    _start_with_digit_or_plus_regex = re.compile(r'^[+\d]')
    _newline_tab_regex = re.compile(r'([\n\t])')
    _exclude_extension_regex = re.compile(r"(#|extension|ext\.?|x\.?)")

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE   ,
            supported_entity: str = "PHONE_NUMBER",
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
        Check if the pattern text cannot be validated as a Phone Number.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        checks = [
            len(self._non_digit_regex.sub("", pattern_text)) < 7,  # Check length after removing non-digit characters
            self._invalid_char_regex.search(self._exclude_extension_regex.sub("", pattern_text)) is not None,  # Check for invalid characters
            self._start_with_digit_or_plus_regex.search(pattern_text) is None,  # Check if it starts with digit/+
            self._newline_tab_regex.search(pattern_text) is not None,  # Check if there's a newline/tab
        ]
        return any(checks)
