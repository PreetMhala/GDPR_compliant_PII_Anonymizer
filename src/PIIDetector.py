from pathlib import Path
import datetime
import uuid
import json
from typing import Union
from .utils.constants import *

__import__('sys').path.append('/'.join(__file__.split('/')[:-1]))  # noqa: E402
try:
    import src.models as pii_models
except ImportError:
    import src.models as pii_models

from src.log_utils import log_event, get_timestamp


def _read_config(config_path: str = "recognizer_en_de_config.yaml") -> dict:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


class PIIDetector:
    def __init__(self, config: Union[str, dict] = None, **kwargs):
        self.trace_id = kwargs.get("traceId", str(uuid.uuid4()))
        self.natco_code = kwargs.get("natco_code", "en")
        self.usecase_version = kwargs.get("usecase_version", "v1.0")
        self.source_intent = kwargs.get("source_intent")

        # Load config (dict or YAML file)
        if isinstance(config, dict):
            self.config = {"SUPPORTED_LANGUAGES": [LANG_CODE], **config}
        elif isinstance(config, str):
            self.config = _read_config(config)
        else:
            config_path = kwargs.pop("config_path", (Path(__file__).parent / GREEK_CONFIG_PATH).resolve().as_posix())
            self.config = _read_config(config_path=config_path)

        self.entities_to_allow = set(e.lower().strip() for e in self.config.get("ENTITIES_TO_ALLOW", [])) \
            if isinstance(config, dict) else set()

        self.supported_languages = kwargs.get("supported_languages", self.config.get("SUPPORTED_LANGUAGES", ["en"]))

        presidio_conf = self.config.get("PRESIDIO") or {}

        if kwargs.get("presidio", "PRESIDIO" in self.config):
            log_event(
                step="Setup",
                event="Loading Presidio",
                message="Initializing Presidio model",
                natco_code=self.natco_code,
                usecase_version=self.usecase_version,
                traceId=self.trace_id,
                model="Presidio",
                source_intent=self.source_intent
            )

            self.presidio_analyzer = pii_models.PresidioAnalyzer(
                supported_languages=self.supported_languages,
                spacy=presidio_conf.get("ENABLE_SPACY", False),
                flair=presidio_conf.get("ENABLE_FLAIR", False),
                cust_patterns=presidio_conf.get("ENABLE_CUSTOM_PATTERNS", False),
                additional_patterns=kwargs.get("additional_patterns"),
                blacklist=kwargs.get("blacklist"),
                whitelist=kwargs.get("whitelist")
            )

            log_event(
                step="Setup",
                event="Presidio loaded",
                message="Presidio detection model ready",
                natco_code=self.natco_code,
                usecase_version=self.usecase_version,
                traceId=self.trace_id,
                model="Presidio",
                source_intent=self.source_intent
            )

    def denonymize(self, presidio_results):
        if hasattr(self, "presidio_analyzer"):
            return self.presidio_analyzer.denonymize(
                anonymized_text=presidio_results.get("anonymized"),
                anonymized_items=presidio_results.get("anonymized_items"),
                entity_mapping=presidio_results.get("entity_mapping")
            ).text
        raise NotImplementedError("Denonymize is not implemented for models other than Presidio.")

    def evaluate(self, text: str, **kwargs) -> dict:
        if hasattr(self, "entities_to_allow"):
            kwargs["allow_list"] = list(self.entities_to_allow)

        results = {}
        for model in ["presidio"]:
            if hasattr(self, f"{model}_analyzer"):
                m_start_time = datetime.datetime.now()

                model_results = getattr(self, f"{model}_analyzer").evaluate(
                    text,
                    **kwargs
                )

                results[model] = model_results

                m_time = datetime.datetime.now() - m_start_time
                log_event(
                    step="Anonymization",
                    event=f"{model.capitalize()} evaluation complete",
                    message=f"{model} processed input in {m_time.total_seconds():.3f} seconds",
                    natco_code=self.natco_code,
                    usecase_version=self.usecase_version,
                    traceId=self.trace_id,
                    model=model.capitalize(),
                    source_intent=self.source_intent
                )
        return results
