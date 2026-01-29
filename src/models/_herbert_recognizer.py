from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation
from typing import Optional, List
from src.models.__base_models import _TransformersNERBaseModel
from src.utils.constants import LANG_CODE  # Typically "pl"


class HerbertBasedModel(_TransformersNERBaseModel):
    def __init__(self, model_name: str = "pczarnik/herbert-base-ner", **kwargs):
        # No special NER_LABELS mapping needed; the pipeline returns standard keys
        if 'aggregation_mechanism' in kwargs:
            pipeline_kwargs = {"aggregation_strategy": kwargs.pop('aggregation_mechanism')}
        else:
            pipeline_kwargs = {"grouped_entities": True}

        pipeline_kwargs.update(kwargs.pop("pipeline_kwargs", {}))

        super().__init__(
            task="token-classification",
            model_name=model_name,
            pipeline_kwargs=pipeline_kwargs,
            tokenizer_kwargs={"use_fast": True, "clean_up_tokenization_spaces": True},
            **kwargs
        )

    def predict(self, text: str, ents: list = None, thres: float = 0.35, **kwargs):
        # Skip base class's rename_keys logic
        results = self.pipe(text)
        return [
            e for e in results
            if (ents is None or e.get("entity") in ents) and e.get("score", 0) > thres
        ]


class _HerbertRec(EntityRecognizer):
    DEFAULT_EXPLANATION = "Identified as {} by Herbert NER model"

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "MISC": "MISC",
    }

    def __init__(
        self,
        supported_language: str = LANG_CODE,
        supported_entities: Optional[List[str]] = None,
        name: Optional[str] = "HerbertRecognizer",
        model_name: str = "pczarnik/herbert-base-ner",
        ner_strength: float = 0.85,
    ):
        self.ner_strength = ner_strength
        self.model = HerbertBasedModel(model_name=model_name)

        supported_entities = supported_entities or list(self.PRESIDIO_EQUIVALENCES.values())
        self._supported_entities_set = set(supported_entities)

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    def load(self) -> None:
        pass  # Already loaded in __init__

    def analyze(self, text, entities, nlp_artifacts=None):
        requested_entities = self._supported_entities_set.intersection(entities)
        if not requested_entities:
            return []

        preds = self.model.predict(text)

        results = []
        for entity in preds:
            # Use 'entity_group' directly, no B-/I- prefix present
            entity_group = entity.get("entity_group", "")

            mapped_label = self.PRESIDIO_EQUIVALENCES.get(entity_group)
            if mapped_label and mapped_label in requested_entities:
                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    textual_explanation=self.DEFAULT_EXPLANATION.format(entity_group),
                    original_score=self.ner_strength,
                )
                results.append(
                    RecognizerResult(
                        entity_type=mapped_label,
                        start=entity.get("start"),
                        end=entity.get("end"),
                        score=self.ner_strength,
                        analysis_explanation=explanation,
                        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name},
                    )
                )
        return results


def get_recognizers(supported_languages: List[str]):
    return [_HerbertRec(supported_language="pl")] if "pl" in supported_languages else []
