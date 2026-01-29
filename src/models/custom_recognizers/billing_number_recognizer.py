from presidio_analyzer import PatternRecognizer, Pattern
from typing import List, Optional
from src.utils.constants import LANG_CODE


class CustBillingNumberRecognizer(PatternRecognizer):
    """
    Recognize Billing Numbers using regex.

    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = LANG_CODE,
            supported_entity: str = "BILLING_NUMBER",
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

