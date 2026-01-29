from presidio_analyzer import PatternRecognizer, Pattern
import re
from typing import List, Optional
from src.utils.constants import LANG_CODE

class NetworkAddressRecognizer(PatternRecognizer):
    """
    Recognize IPv4, IPv6, and MAC addresses using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    # Precompiled regex for performance
    _ipv4_regex = re.compile(r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
    _ipv6_regex = re.compile(r"\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
    _mac_colon_regex = re.compile(r"\b([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
    _mac_hyphen_regex = re.compile(r"\b([0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\b")
    _mac_no_separator_regex = re.compile(r"\b[0-9a-fA-F]{12}\b")

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE        ,
            supported_entity: str = "NETWORK_ADDRESS",
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
        Check if the detected IP or MAC address is invalid.

        :param pattern_text: Text detected as pattern by regex
        :return: True if invalid, False if valid
        """
        checks = [
            self._ipv4_regex.match(pattern_text) is None,  # For IPv4
            self._ipv6_regex.match(pattern_text) is None,  # For IPv6
            self._mac_colon_regex.match(pattern_text) is None,  # For colon-separated MAC
            self._mac_hyphen_regex.match(pattern_text) is None,  # For hyphen-separated MAC
            self._mac_no_separator_regex.match(pattern_text) is None,  # For no-separator MAC
        ]
        return any(checks)
