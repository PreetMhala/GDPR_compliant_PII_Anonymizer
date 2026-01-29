import datetime
import logging
from typing import Optional, List, Tuple, Set
from presidio_analyzer import RecognizerResult, EntityRecognizer, AnalysisExplanation
from src.utils import SPACY_MODELS
from src.utils.constants import LANG_CODE
from gr_nlp_toolkit import Pipeline


logger = logging.getLogger(__name__)


class _SpacyRec(EntityRecognizer):
    DEFAULT_EXPLANATION = "Identified as {} by Spacy's Named Entity Recognition"

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION"}),
        ({"STREET_ADDRESS"}, {"STREET_ADDRESS"}),
        ({"COORDINATE"}, {"COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"ORGANIZATION"}, {"ORG"}),
    ]

    def __init__(
            self,
            supported_language: str = LANG_CODE,
            supported_entities: Optional[List[str]] = None,
            check_label_groups: Optional[Tuple[Set, Set]] = None,
            ner_strength: float = 0.85,
            name: Optional[str] = None,
    ):
        self.ner_strength = ner_strength

        self.check_label_groups = check_label_groups or self.CHECK_LABEL_GROUPS
        self._entity_label_map = self._build_entity_label_map(self.check_label_groups)

        if supported_language not in SPACY_MODELS:
            raise ValueError(f"Language must be one of: {list(SPACY_MODELS.keys())}")

        spacy_model = SPACY_MODELS[supported_language]
        logger.info(f"[SPACY] Loading Spacy model - {spacy_model}")

        supported_entities = supported_entities or [
            "LOCATION", "PERSON", "NRP", "ORGANIZATION", "STREET_ADDRESS", "COORDINATE"
        ]
        self._supported_entities_set = set(supported_entities)

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name
        )

    def load(self) -> None:
        """Model is already loaded at initialization."""
        pass

    def analyze(self, text1, entities, nlp_artifacts=None):  # noqa: D102

        if not nlp_artifacts:
            logger.warning("[SPACY] Skipping SpaCy, nlp artifacts not provided...")
            return []

        results = []
        ner_entities = nlp_artifacts.entities
        requested_entities = self._supported_entities_set.intersection(entities)

        a_start_time = datetime.datetime.now()

        for ent in ner_entities:
            label = ent.label_
            for entity in requested_entities:
                valid_labels = self._entity_label_map.get(entity)
                if valid_labels and label in valid_labels:
                    explanation = AnalysisExplanation(
                        self.__class__.__name__,
                        self.ner_strength,
                        textual_explanation=self.DEFAULT_EXPLANATION.format(label),
                    )
                    results.append(
                        RecognizerResult(
                            entity_type=entity,
                            start=ent.start_char,
                            end=ent.end_char,
                            score=self.ner_strength,
                            analysis_explanation=explanation,
                            recognition_metadata={
                                RecognizerResult.RECOGNIZER_NAME_KEY: self.name
                            },
                        )
                    )
        _ = datetime.datetime.now() - a_start_time  # retained for possible future timing/debug
        return results

    @staticmethod
    def _build_entity_label_map(label_groups: List[Tuple[Set[str], Set[str]]]) -> dict:
        """Convert CHECK_LABEL_GROUPS into a fast lookup map."""
        entity_to_labels = {}
        for entity_group, label_group in label_groups:
            for entity in entity_group:
                entity_to_labels.setdefault(entity, set()).update(label_group)
        return entity_to_labels
