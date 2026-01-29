import time
import logging
from typing import Optional, List, Set, Dict
from src.utils.constants import LANG_CODE
from gr_nlp_toolkit import Pipeline

from src.utils import FLAIR_MODELS
from presidio_analyzer import RecognizerResult, EntityRecognizer, AnalysisExplanation
from presidio_analyzer.nlp_engine import NlpArtifacts

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ImportError:
    raise ImportError("Flair is not installed")


logger = logging.getLogger("pii-identifier")


class _FlairRec(EntityRecognizer):
    ENTITIES = [
        "LOCATION", "PERSON", "NRP", "ORGANIZATION", "GPE",
        "MAC_ADDRESS", "US_BANK_NUMBER", "IMEI", "LICENSE_PLATE",
        "PASSPORT", "CURRENCY", "ROUTING_NUMBER", "US_ITIN",
        "US_DRIVER_LICENSE", "AGE", "PASSWORD", "SWIFT_CODE",
        "STREET_ADDRESS", "COORDINATE",
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "NORP": "NRP",
        "NROP": "NRP",
        "US_PASSPORT": "PASSPORT",
        # ... add others as needed
    }

    def __init__(
            self,
            supported_language: str = LANG_CODE,
            supported_entities: Optional[List[str]] = None,
            model: SequenceTagger = None,
            name: Optional[str] = None,
    ):
        # Initialize supported_entities and label mappings
        supported_entities = supported_entities or self.ENTITIES
        super().__init__(supported_entities, name, supported_language)

        if supported_language not in FLAIR_MODELS:
            raise ValueError(f"Language must be one of {list(FLAIR_MODELS.keys())}")

        logger.info(f"[FLAIR] Loading Flair model - {FLAIR_MODELS[supported_language]}")
        self.model = model or SequenceTagger.load(FLAIR_MODELS[supported_language])

        # Precompute flair_label -> presidio mapping and explanation cache
        self._label_map: Dict[str, str] = {}
        for flair_label, presidio_label in self.PRESIDIO_EQUIVALENCES.items():
            if presidio_label in self.supported_entities:
                self._label_map[flair_label] = presidio_label
        self._explanation_cache: Dict[str, AnalysisExplanation] = {}

    def analyze(
            self,
            text: str,
            entities: Optional[List[str]] = None,
            nlp_artifacts: NlpArtifacts = None,
    ) -> List[RecognizerResult]:
        """
        Analyze text using Flair NER with optimized single-pass processing.
        """


        start_time = time.perf_counter()

        requested = set(entities) if entities else set(self.supported_entities)
        sentence = Sentence(text)
        self.model.predict(sentence)

        results: List[RecognizerResult] = []
        for span in sentence.get_spans("ner"):
            flair_label = span.labels[0].value
            # Map Flair label to Presidio type
            presidio_type = self._label_map.get(flair_label)
            if not presidio_type or presidio_type not in requested:
                continue

            # Cache or build explanation
            if presidio_type not in self._explanation_cache:
                explanation_text = self.DEFAULT_EXPLANATION.format(flair_label)
                self._explanation_cache[presidio_type] = AnalysisExplanation(
                    self.__class__.__name__, 0.0, textual_explanation=explanation_text
                )
            explanation = self._explanation_cache[presidio_type]

            score = round(span.score, 2)
            # Update explanation score dynamically
            explanation.score = score

            result = RecognizerResult(
                entity_type=presidio_type,
                start=span.start_position,
                end=span.end_position,
                score=score,
                analysis_explanation=explanation,
            )
            results.append(result)

        elapsed = time.perf_counter() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "-- [FLAIR] analyzed %d spans in %.4f seconds",
                len(results), elapsed,
            )

        return results
