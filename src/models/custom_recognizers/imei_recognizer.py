from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
import re
from src.utils.constants import LANG_CODE


class CustIMEIRecognizer(PatternRecognizer):
    """
    Recognize IMEI numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex patterns for performance
    _non_digit_regex = re.compile(r"\D")
    _separator_regex = re.compile(r'(\d{2,})\.')
    _all_zero_regex = re.compile(r'^[0]{15}$')

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE        ,
            supported_entity: str = "IMEI",
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
        Check if the pattern text cannot be validated as a valid IMEI number.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalid, False if valid
        """
        checks = [
            len(self._non_digit_regex.sub("", pattern_text)) != 15,  # not 15 digits
            self._separator_regex.search(pattern_text) is not None,  # not valid separators
            self._all_zero_regex.match(self._non_digit_regex.sub("", pattern_text)) is not None,  # all zeros
        ]
        return any(checks)
