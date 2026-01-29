import re
import logging
from typing import List, Optional
from presidio_analyzer import EntityRecognizer, RecognizerResult
from gr_nlp_toolkit import Pipeline
from src.utils.constants import LANG_CODE

logger = logging.getLogger(__name__)


class GRNLPToolkitRecognizer(EntityRecognizer):
    """
    A recognizer that uses the GR NLP Toolkit for Greek NER tasks.
    """

    def __init__(self, supported_language: str = LANG_CODE, name: str = "GR_NLP_TOOLKIT"):
        try:
            self.pipeline = Pipeline("ner")
            logger.info("[GR_NLP] GR NLP Toolkit pipeline loaded successfully.")
        except Exception:
            logger.exception("[GR_NLP] Failed to initialize GR NLP Toolkit pipeline.")
            self.pipeline = None

        self.label_map = {
            "PERSON": "PERSON",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "FAC": "LOCATION",
            "ORG": "ORGANIZATION",
            "MISC": "MISC",
            "DATE": "DATE"
        }

        super().__init__(supported_entities=[], supported_language=supported_language, name=name)

    def load(self) -> bool:
        return self.pipeline is not None

    def analyze(
            self,
            text: str,
            entities: Optional[List[str]] = None,
            nlp_artifacts=None
    ) -> List[RecognizerResult]:
        if not self.pipeline:
            logger.warning("[GR_NLP] GR NLP pipeline not loaded.")
            return []
        results: List[RecognizerResult] = []

        ner_pipeline = Pipeline("ner")
        doc = ner_pipeline(text)
        tokens = doc.tokens

        # Build token offsets using re.finditer()
        token_offsets = []
        search_pos = 0

        for idx, token in enumerate(tokens):
            token_text = token.text
            match = re.search(re.escape(token_text), text[search_pos:])
            if match:
                start = search_pos + match.start()
                end = search_pos + match.end()
                token_offsets.append((start, end))
                search_pos = end  # Move forward
            else:
                token_offsets.append((-1, -1))  # Unmatched fallback
                logger.warning(f"[GR_NLP] Token not found in text: {token_text}")

        # NER tag sequence parsing (B/I/E/S)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            ner_tag = token.ner
            norm_start, norm_end = token_offsets[i]

            if norm_start == -1:
                i += 1
                continue

            if ner_tag.startswith("B-"):
                entity_type = ner_tag[2:]
                start_token_index = i
                end_token_index = i
                i += 1

                while i < len(tokens):
                    next_ner = tokens[i].ner
                    if next_ner.startswith("I-") or next_ner.startswith("E-"):
                        end_token_index = i
                        if next_ner.startswith("E-"):
                            i += 1
                            break
                        i += 1
                    else:
                        break

                span_start = token_offsets[start_token_index][0]
                span_end = token_offsets[end_token_index][1]
                matched_text = text[span_start:span_end]
                entity = self.label_map.get(entity_type, entity_type)

                results.append(RecognizerResult(
                    entity_type=entity,
                    start=span_start,
                    end=span_end,
                    score=0.9,
                    analysis_explanation=None
                ))

            elif ner_tag.startswith("S-"):
                entity_type = ner_tag[2:]
                entity = self.label_map.get(entity_type, entity_type)
                matched_text = text[norm_start:norm_end]
                results.append(RecognizerResult(
                    entity_type=entity,
                    start=norm_start,
                    end=norm_end,
                    score=0.9,
                    analysis_explanation=None
                ))
                i += 1
            else:
                i += 1

        return results


def get_recognizers(supported_languages: List[str]):
    if "el" not in supported_languages:
        return []
    return [GRNLPToolkitRecognizer(supported_language=LANG_CODE)]
