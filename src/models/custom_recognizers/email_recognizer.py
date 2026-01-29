from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
import re
from src.utils.constants import LANG_CODE


class CustEmailRecognizer(PatternRecognizer):
    """
    Recognize Email using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    _compiled_checks = [
        re.compile(r'[.%+-]{2,}'),                      # Check multiple special characters together
        re.compile(r'^[^@]+@[^@]+$'),                   # Must have one @
        re.compile(r'(?<=@)[^@]+(?=\.[a-z]{2,})'),      # Domain pattern
        re.compile(r'\.com\.\w{2,}$'),                  # Bad TLD pattern
        re.compile(r'(\d{2,})\.'),                      # Numeric dot patterns
        re.compile(r'^[^@]+\.[^@]{2,}$'),               # Simple validation
    ]

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE        ,
            supported_entity: str = "EMAIL_ADDRESS",
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
        Check if the pattern text cannot be validated as a Email.

        :param pattern_text: Text detected as pattern by regex
        :return: True if validated
        """
        checks = [
            regex.search(pattern_text) is not None
            for regex in self._compiled_checks
        ]
        return all(checks)
