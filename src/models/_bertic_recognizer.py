from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation
import datetime
from src.models.__base_models import _TransformersNERBaseModel
from typing import Optional, List
from src.utils.constants import LANG_CODE

class BERTicBasedModel(_TransformersNERBaseModel):
    def __init__(self, model_name: str = "classla/bcms-bertic-ner", **kwargs):
        ner_labels = kwargs.pop("NER_LABELS", {"entity": "entity_group"})
        if 'aggregation_mechanism' in kwargs:
            pipeline_kwargs = {"aggregation_strategy": kwargs.pop('aggregation_mechanism')}
        else:
            pipeline_kwargs = {"grouped_entities": True}

        pipeline_kwargs = {**pipeline_kwargs, **kwargs.pop("pipeline_kwargs", {})}
        super().__init__(
            task="token-classification",
            model_name=model_name,
            NER_LABELS=ner_labels,
            pipeline_kwargs=pipeline_kwargs,
            tokenizer_kwargs={"use_fast": True, "clean_up_tokenization_spaces": True},
            **kwargs
        )

    def predict(self, text: str):
        return self.pipe(text)

class _BerticRec(EntityRecognizer):
    DEFAULT_EXPLANATION = "Identified as {} by BERTic NER model"

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "MISC": "MISC"
    }

    def __init__(
        self,
        supported_language: str = LANG_CODE,
        supported_entities: Optional[List[str]] = None,
        name: Optional[str] = "BERTICRecognizer",
        model_name: str = "classla/bcms-bertic-ner",
        ner_strength: float = 0.85,
    ):
        self.ner_strength = ner_strength
        self.model = BERTicBasedModel(model_name=model_name)

        supported_entities = supported_entities or list(set(self.PRESIDIO_EQUIVALENCES.values()))
        self._supported_entities_set = set(supported_entities)

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    def load(self) -> None:
        # Already loaded in init
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        requested_entities = self._supported_entities_set.intersection(entities)
        if not requested_entities:
            return []

        preds = self.model.predict(text)
        results = []
        for entity in preds:
            bertic_label = entity.get("entity_group")
            start = entity.get("start")
            end = entity.get("end")

            mapped_label = self.PRESIDIO_EQUIVALENCES.get(bertic_label)
            if mapped_label and mapped_label in requested_entities:
                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    textual_explanation=self.DEFAULT_EXPLANATION.format(bertic_label),
                    original_score=self.ner_strength,
                )
                results.append(
                    RecognizerResult(
                        entity_type=mapped_label,
                        start=start,
                        end=end,
                        score=self.ner_strength,
                        analysis_explanation=explanation,
                        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name},
                    )
                )
        return results

def get_recognizers(supported_languages: List[str]):
    if "hr" not in supported_languages and "cnr" not in supported_languages:
        return []
    return [_BerticRec(supported_language=LANG_CODE)]

