import re
from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
from src.utils.constants import LANG_CODE


class CustUSSsnRecognizer(PatternRecognizer):
    """Recognize US Social Security Number (SSN) using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex patterns for performance
    _invalid_start_regex = re.compile(r'^(?:000|666|9[0-9]{2})\d{6}$')
    _invalid_middle_group_regex = re.compile(r'^\d{3}00\d{4}')
    _invalid_last_group_regex = re.compile(r'^\d{5}0000')
    _invalid_start_with_digit_regex = re.compile(r'^[^0-9]')
    _same_digit_check = re.compile(r'^(\d)\1*$')  # check if all digits are the same
    _all_digit_check = re.compile(r'\d{9}')  # Ensure it's 9 digits

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE ,
            supported_entity: str = "SSN",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        """
        Check if the pattern text cannot be validated as a US_SSN entity.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalidated
        """
        # Normalize the input pattern text
        for char, repl in zip('iIlZEASBO', '111234580'):
            pattern_text = pattern_text.replace(char, repl)

        # Remove non-numeric characters except 0-9
        digits = re.sub('[^1-9]', '', pattern_text)

        # Check length is exactly 9
        if len(digits) != 9:
            return False

        # Validate invalid SSN patterns
        checks = [
            self._invalid_start_regex.match(digits),
            self._invalid_middle_group_regex.match(digits),
            self._invalid_last_group_regex.match(digits),
            self._invalid_start_with_digit_regex.match(pattern_text),
            len(set(digits)) == 1,  # all digits the same
            self._same_digit_check.match(digits),  # all digits the same
            digits[3:5] == "00",  # middle group cannot be 00
            digits[5:] == "0000",  # last group cannot be 0000
        ]

        return any(checks)


class CustSsnRecognizer(PatternRecognizer):
    """Recognize General Social Security Number (SSN) using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE ,
            supported_entity: str = "SSN",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
