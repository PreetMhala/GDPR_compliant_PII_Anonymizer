from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation
from src.models.__base_models import _TransformersNERBaseModel
from typing import Optional, List
from src.utils.constants import LANG_CODE

class NerkorBasedModel(_TransformersNERBaseModel):
    def __init__(self, model_name: str = "novakat/nerkor-cars-onpp-hubert", **kwargs):
        ner_labels = kwargs.pop("NER_LABELS", {"entity": "entity_group"})

        # Use grouped_entities like BERTic
        if 'aggregation_mechanism' in kwargs:
            pipeline_kwargs = {"aggregation_strategy": kwargs.pop("aggregation_mechanism")}
        else:
            pipeline_kwargs = {"grouped_entities": True}

        pipeline_kwargs.update(kwargs.pop("pipeline_kwargs", {}))

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



class _NerkorRec(EntityRecognizer):
    DEFAULT_EXPLANATION = "Identified as {} by Nerkor NER model"

    PRESIDIO_EQUIVALENCES = {
        # Standard OntoNotes-style mappings
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "FAC": "LOCATION",
        "NORP": "MISC",
        "LANGUAGE": "MISC",
        "PRODUCT": "MISC",
        "PROD": "MISC",
        "WORK_OF_ART": "MISC",
        "LAW": "MISC",
        "EVENT": "MISC",

        # Numeric/time values
        "DATE": "DATE",
        "TIME": "TIME",
        "DURATION": "TIME",
        "DUR": "TIME",
        "AGE": "DATE",
        "ORDINAL": "NUMBER",
        "CARDINAL": "NUMBER",
        "PERCENT": "NUMBER",
        "MONEY": "MONEY",
        "QUANTITY": "NUMBER",

        # Additional useful categories from their extended scheme
        "ID": "ID",
        "AWARD": "MISC",
        "CAR": "PRODUCT",
        "MEDIA": "ORG",
        "SMEDIA": "ORG",
        "PROJ": "MISC",
        "MISC": "MISC",
        "MISC-ORG": "ORGANIZATION",
    }

    def __init__(
        self,
        supported_language: str = LANG_CODE,
        supported_entities: Optional[List[str]] = None,
        name: Optional[str] = "NerkorRecognizer",
        model_name: str = "novakat/nerkor-cars-onpp-hubert",
        ner_strength: float = 0.85,
    ):
        self.ner_strength = ner_strength
        self.model = NerkorBasedModel(model_name=model_name)

        supported_entities = supported_entities or list(set(self.PRESIDIO_EQUIVALENCES.values()))
        self._supported_entities_set = set(supported_entities)

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

    def load(self) -> None:
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        requested_entities = self._supported_entities_set.intersection(entities)
        if not requested_entities:
            return []

        preds = self.model.predict(text)
        results = []
        for entity in preds:
            label = entity.get("entity_group")
            start = entity.get("start")
            end = entity.get("end")

            mapped_label = self.PRESIDIO_EQUIVALENCES.get(label)
            if mapped_label and mapped_label in requested_entities:
                explanation = AnalysisExplanation(
                    recognizer=self.name,
                    textual_explanation=self.DEFAULT_EXPLANATION.format(label),
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
    if "hu" not in supported_languages:
        return []
    return [_NerkorRec(supported_language=LANG_CODE)]
