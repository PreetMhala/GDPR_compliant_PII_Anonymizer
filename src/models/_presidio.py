import re
from functools import lru_cache
from src.utils import model_loader
import yaml
from pathlib import Path
import logging
from typing import Dict, List
from src.utils.constants import *
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from gr_nlp_toolkit import Pipeline as GRPipeline
from presidio_anonymizer.entities import OperatorConfig, OperatorResult
from src.utils.model_loader import discover_model_recognizers
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    RecognizerResult,
    Pattern,
    PatternRecognizer,
    LocalRecognizer,
)
from presidio_anonymizer.operators import Operator, OperatorType
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

from src.models.custom_recognizers import (
    CustPhoneNumberRecognizer,
    CustEmailRecognizer,
    CustGenericIDRecognizer,
    CustUrlRecognizer,
    CustBillingNumberRecognizer,
    CustIPAddressRecognizer,
    CustSsnRecognizer,
    CustUSSsnRecognizer,
    CustCreditCardRecognizer,
    CustIMEIRecognizer,
    NetworkAddressRecognizer,
    CustIbanRecognizer,
    CustIanRecognizer,
)
from src.models.__spacy_recognizer import _SpacyRec
from src.models.__flair_recognizer import _FlairRec
from src.utils import SPACY_MODELS, FLAIR_MODELS
from src.utils.pattern_context_loader import pattern_context_read

__import__("os").environ["TOKENIZERS_PARALLELISM"] = "false"
__import__("warnings").filterwarnings('ignore')
logger = logging.getLogger("pii-identifier")


class CustomContextEnhancer(LemmaContextAwareEnhancer):
    def __init__(
            self,
            context_similarity_factor: float = 0.35,
            min_score_with_context_similarity: float = 0.7,
            context_prefix_count: int = 10,
            context_suffix_count: int = 5,
    ):
        super().__init__(
            context_similarity_factor=context_similarity_factor,
            min_score_with_context_similarity=min_score_with_context_similarity,
            context_prefix_count=context_prefix_count,
            context_suffix_count=context_suffix_count,
        )

    @staticmethod
    def _find_supportive_word_in_context(
            context_list: List[str], recognizer_context_list: List[str]
    ) -> str:
        if not context_list or not recognizer_context_list:
            return ""
        for keyword in recognizer_context_list:
            if any(keyword.lower() in word.lower() for word in context_list):
                logger.debug("Found context keyword '%s'", keyword)
                return keyword
        return ""


class InstanceCounterAnonymizer(Operator):
    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: Dict = None) -> str:
        entity_type = params["entity_type"]
        mapping = params["entity_mapping"]
        type_map = mapping.setdefault(entity_type, {})
        if text in type_map:
            return type_map[text]
        index = max((int(val.split('_')[-1][:-1]) for val in type_map.values()), default=-1) + 1
        placeholder = self.REPLACING_FORMAT.format(entity_type=entity_type, index=index)
        type_map[text] = placeholder
        return placeholder

    def validate(self, params: Dict = None) -> None:
        if "entity_mapping" not in params or "entity_type" not in params:
            raise ValueError("Parameters 'entity_mapping' and 'entity_type' are required.")

    def operator_name(self) -> str:
        return ENTITY_COUNTER

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class InstanceCounterDeanonymizer(Operator):
    def operate(self, text: str, params: Dict = None) -> str:
        entity_type = params[ENTITY_TYPE]
        mapping = params[ENTITY_MAPPING]
        if entity_type not in mapping or text not in mapping[entity_type].values():
            raise ValueError(f"Invalid entity or placeholder: {entity_type}, {text}")
        return next(k for k, v in mapping[entity_type].items() if v == text)

    def validate(self, params: Dict = None) -> None:
        if "entity_mapping" not in params or "entity_type" not in params:
            raise ValueError("Parameters 'entity_mapping' and 'entity_type' are required.")

    def operator_name(self) -> str:
        return "entity_counter_deanonymizer"

    def operator_type(self) -> OperatorType:
        return OperatorType.Deanonymize


class PresidioAnalyzer:
    DEFAULT_ADDITIONAL_RECOGNIZER_CLASSES = [
        (CustPhoneNumberRecognizer, PHONE_RECOGNIZER, CUSTOM_PHONE, SHARED_KEYWORD_PHONENUMBER),
        (CustIanRecognizer, IAN_RECOGNIZER, CUSTOM_IAN, SHARED_KEYWORD_IAN),
        (CustIMEIRecognizer, IMEI_RECOGNIZER, CUSTOM_IMEI, SHARED_KEYWORD_IMEI),
        (CustIbanRecognizer, IBAN_RECOGNIZER, CUSTOM_IBAN, SHARED_KEYWORD_IBAN),
        (CustBillingNumberRecognizer, BILLING_NUMBER_RECOGNIZER, CUSTOM_BILLING_NUMBER, SHARED_KEYWORD_BILLING),
        (NetworkAddressRecognizer, NETWORD_ADDRESS_RECOGNIZER, CUSTOM_NETWORD_ADDRESS, SHARED_KEYWORD_NETWORK),
        (CustEmailRecognizer, EMAIL_RECOGNIZER, CUSTOM_EMAIL, SHARED_KEYWORD_EMAIL),
        (CustGenericIDRecognizer, ID_RECOGNIZER, CUSTOM_ID, SHARED_KEYWORD_ID),
        (CustUrlRecognizer, URL_RECOGNIZER, CUSTOM_URL, SHARED_KEYWORD_URL),
        (CustSsnRecognizer, SSN_RECOGNIZER, CUSTOM_SSN, SHARED_KEYWORD_SSN),
        (CustUSSsnRecognizer, USSSN_RECOGNIZER, CUSTOM_USSSN, SHARED_KEYWORD_USSSN),
        (CustCreditCardRecognizer, CREDIT_CARD_RECOGNIZER, CUSTOM_CREDIT_CARD, SHARED_KEYWORD_CREDITCARD),
        (CustIPAddressRecognizer, IP_ADDR_RECOGNIZER, CUSTOM_IP_ADDR, SHARED_KEYWORD_IP),
    ]

    def __init__(
            self,
            supported_languages: List[str] = None,
            spacy: bool = True,
            flair: bool = True,
            cust_patterns: bool = True,
            additional_patterns: List[tuple] = None,
            blacklist: List[str] = None,
            whitelist: List[str] = None,
            additional_recognizers: List[LocalRecognizer] = None,
    ):
        supported_languages = supported_languages or [LANG_CODE]
        self.supported_languages = supported_languages
        self.whitelist = whitelist or []

        # Load and cache config.      encoding -> utf-8 for handling multilingual text input non english mainly
        with open(GREEK_CONFIG_PATH, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f)
        self._numerical_placeholders = self._config_data.get(NUMERICAL_PLACEHOLDER, {})
        # Precompile placeholder regexes
        self._placeholder_patterns = {
            entity: re.compile(fr"<{entity}_\d+>")
            for entity in self._numerical_placeholders
        }

        self.detector_engine = None

        # Load model-based recognizers
        self.custom_model_recognizers = model_loader.discover_model_recognizers(self.supported_languages)

        # Registry and patterns
        self.registry = RecognizerRegistry(supported_languages=self.supported_languages, global_regex_flags=90)
        if cust_patterns:
            from ..utils.custom_regex_loader import load_patterns_from_yaml
            patterns = load_patterns_from_yaml(GREEK_CONFIG_PATH)
            for p in patterns + (additional_patterns or []):
                try:
                    self.__get_pattern_recognizer(*p)
                except Exception:
                    logger.exception("Error adding custom pattern %s", p)

        # Additional recognizers
        all_recognizers = [*(additional_recognizers or []), *self.__get_default_additional_recognizers()]
        for rec in all_recognizers:
            try:
                self.registry.add_recognizer(rec)
            except Exception:
                logger.exception("Error adding recognizer %s", rec)

        if blacklist:
            self.registry.add_recognizer(PatternRecognizer(supported_entity="BLACKLISTED", deny_list=blacklist))

        # Engines
        self.analyzer_engine = self.__load_analyzer_engine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(InstanceCounterAnonymizer)
        self.deanonymizer_engine = DeanonymizeEngine()
        self.deanonymizer_engine.add_deanonymizer(InstanceCounterDeanonymizer)

    def _load_model_recognizers(self, spacy: bool = True, flair: bool = True):
        # Create a temporary mapping for model loading
        temp_supported_languages = [
            "hr" if lang == "cnr" else lang for lang in self.supported_languages
        ]

        if spacy:
            logger.info("[PRESIDIO] Loading spacy models.")

            # Determine which languages have spacy models available
            spacy_loaded_languages = [
                lang for lang in self.supported_languages
                if (lang == "cnr" and "hr" in SPACY_MODELS) or (lang in SPACY_MODELS)
            ]

            # Load spacy recognizers for the available languages
            self.spacy_recognizers = [
                _SpacyRec(lang, name=SPACY_RECOGNIZER)
                for lang in spacy_loaded_languages
            ]

        if flair:
            logger.info("[PRESIDIO] Loading flair models.")

            # Determine which languages have flair models available
            flair_loaded_languages = [
                lang for lang in self.supported_languages if lang in FLAIR_MODELS
            ]

            # Load flair recognizers for the available languages
            self.flair_recognizers = [
                _FlairRec(supported_language=lang, name=FLAIR_RECOGNIZER)
                for lang in flair_loaded_languages
            ]

        # Always pass the original supported_languages to custom model discovery
        try:
            self.custom_model_recognizers = discover_model_recognizers(self.supported_languages)
            logger.info("[PRESIDIO] Loaded %d dynamic model recognizers.", len(self.custom_model_recognizers))
        except Exception:
            logger.exception("[PRESIDIO] Error loading dynamic model recognizers.")
            self.custom_model_recognizers = []

    def __get_default_additional_recognizers(self) -> List[PatternRecognizer]:
        data = self._config_data
        # Titles
        titles = data[TITLE][TITLES_LIST]
        title_patterns = []
        for e in data[TITLE][TITLE_PATTERNS]:
            regex = e[REGEX].replace("LIST_TITLE", "|".join(re.escape(t) for t in titles))
            title_patterns.append(Pattern(name=e[NAME], regex=regex, score=e[CONFIDENCE]))
        title_context = data.get(TITLE_CONTEXT, [])

        zipcode_patterns = [Pattern(name=e[NAME], regex=e[REGEX], score=e[CONFIDENCE]) for e in data[ZIP][ZIP_PATTERN]]
        zipcode_context = data.get(ZIP_CONTEXT, [])

        passport_context = data.get(PASSPORT_CONTEXT, [])
        passport_data = data[PASSPORT_LIST][PASSPORTS]
        countries = [
            "INDIAN", "US", "ITALY", "CANADA", "FRANCE", "GERMAN", "SWEDEN", "UK", "AUSTRIA", "RUSSIAN"
        ]
        passport_patterns = {
            country: [Pattern(name=e[NAME], regex=e[REGEX], score=e[CONFIDENCE]) for e in passport_data.get(country, [])]
            for country in countries
        }
        pattern_recognizer_args = [
            ("CustTitlesRecognizer", "TITLE", title_patterns, title_context),
            ("CustZipCodeRecognizer", "ZIP_CODE", zipcode_patterns, zipcode_context),
            *[(f"{country}CustPassportRecognizer", f"{country}_PASSPORT", patterns, passport_context) for country, patterns in passport_patterns.items()]
        ]

        # Custom class contexts
        cls_context = []
        for cls, key, ent, ctx_key in self.DEFAULT_ADDITIONAL_RECOGNIZER_CLASSES:
            pats, ctx = pattern_context_read(key, ent, ctx_key)
            cls_context.append((cls, pats, ctx))

        recognizers: List[LocalRecognizer] = []
        for name, ent, pats, ctx in pattern_recognizer_args:
            for lang in self.supported_languages:
                mapped_lang = "hr" if lang == "cnr" else lang
                recognizers.append(
                    PatternRecognizer(supported_entity=ent, patterns=pats, context=ctx, supported_language=mapped_lang)
                )

        return recognizers

    def _map_lang(self, lang: str) -> str:
        return "hr" if lang == "cnr" else lang

    def __get_pattern_recognizer(
            self, supported_entity: str, regex: str, score: float = 0.5,
            context_words: List[str] = None, _add: bool = True
    ):
        pattern = Pattern(name=f"Cust{supported_entity}_pattern", regex=regex, score=score)
        recs: List[PatternRecognizer] = []
        for lang in self.supported_languages:
            mapped_lang = self._map_lang(lang)  # ✅ Use your mapping helper here
            rec = PatternRecognizer(
                supported_entity=supported_entity,
                patterns=[pattern],
                context=context_words,
                supported_language=mapped_lang
            )
            if _add:
                self.registry.add_recognizer(rec)
            recs.append(rec)
        return recs

    @lru_cache(maxsize=1)
    def __load_analyzer_engine(self) -> AnalyzerEngine:
        logger.info("[PRESIDIO] Loading analyzer engine.")

        analyzer_kwargs: Dict = {
            "supported_languages": self.supported_languages,
            "context_aware_enhancer": CustomContextEnhancer(),
        }

        # Map 'cnr' -> 'hr' when looking up Spacy models
        temp_langs = [self._map_lang(lang) for lang in self.supported_languages]

        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": SPACY_MODELS[lang]} for lang in temp_langs if
                       lang in SPACY_MODELS]
        }

        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

        # ✅ Use mapped languages here
        self.registry.load_predefined_recognizers(languages=temp_langs, nlp_engine=nlp_engine)

        # Add your custom recognizers
        for rec in getattr(self, "spacy_recognizers", []):
            self.registry.add_recognizer(rec)
        for rec in getattr(self, "flair_recognizers", []):
            self.registry.add_recognizer(rec)
        for rec in getattr(self, "custom_model_recognizers", []):
            self.registry.add_recognizer(rec)

        # Optionally remove default recognizers you don’t need
        for rec_name in [
            "SpacyRecognizer", URL_RECOG, INPASSPORTRECOG, ITPASSPORTRECOG,
            USPASSPORTRECOG, USSSNRECOG, CREDITCARDRECOG,
        ]:
            try:
                self.registry.remove_recognizer(rec_name)
            except Exception:
                pass

        analyzer_kwargs[NLP_ENGINE] = nlp_engine
        return AnalyzerEngine(**analyzer_kwargs, registry=self.registry)

    def _analyze(self, text: str, ents: List[str] = None, thres: float = 0.35, **kwargs) -> List[RecognizerResult]:
        language = kwargs.get("language")

        # If language is a list, take the first one
        if isinstance(language, list):
            language = language[0]

        # Validate the language against supported languages
        if language not in self.supported_languages:
            logger.warning("[PRESIDIO] Unsupported language '%s', trying fallback detector.", language)
            language = None

        # If no language, try to auto-detect
        if language is None and self.detector_engine:
            language = self.detector_engine.evaluate(text).get("default", [LANG_CODE])[0]

        # ✅ Map 'cnr' → 'hr' when passing to the NLP engine
        language_for_nlp = "hr" if language == "cnr" else language

        # Process entity filter
        ents = None if ents and "All" in ents else ents

        return self.analyzer_engine.analyze(
            text=text,
            entities=ents,
            score_threshold=thres,
            allow_list=[*self.whitelist, *(kwargs.get("allow_list") or [])],
            allow_list_match=("regex" if kwargs.get("allow_list_match") == "regex" else "exact"),
            language=language_for_nlp,  # 👈 Pass mapped language to NLP
            return_decision_process=kwargs.get("return_decision_process", False),
        )

    from presidio_analyzer import RecognizerResult
    from typing import List, Dict

    def _anonymize(self, text: str, analyze_res: List[RecognizerResult] = None, allow_list: List[str] = None, **kwargs):
        if allow_list is None:
            allow_list = []

        entity_mapping: Dict = {}

        analyze_res = analyze_res or self._analyze(text, **kwargs)

        # Filter entities to exclude those in allow_list
        filtered_entities = []
        for entity in analyze_res:
            entity_text = text[entity.start:entity.end]
            # Case-insensitive check (optional), adjust as needed
            if entity_text.lower() not in (s.lower() for s in allow_list):
                filtered_entities.append(entity)
            else:
                # Optionally log skipping
                # print(f"Skipping anonymization for allowed entity: {entity_text}")
                pass

        results = self.anonymizer_engine.anonymize(
            text,
            filtered_entities,
            {"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": entity_mapping})}
        )

        return results, entity_mapping

    def denonymize(self, anonymized_text: str, anonymized_items: List[OperatorResult], entity_mapping: Dict):
        return self.deanonymizer_engine.deanonymize(
            anonymized_text,
            anonymized_items,
            {"DEFAULT": OperatorConfig("entity_counter_deanonymizer", {"entity_mapping": entity_mapping})}
        )

    def replace_numerical_placeholders(self, anonymized_text: str) -> str:
        for entity, placeholder in self._numerical_placeholders.items():
            pattern = self._placeholder_patterns[entity]
            anonymized_text = pattern.sub(placeholder, anonymized_text)
        return anonymized_text

    def evaluate(self, text: str, **kwargs):
        allow_list = kwargs.pop("allow_list", [])

        original_text = text
        if LANG_CODE == "el":
            g2g = GRPipeline("g2g")
            normalized_text = g2g(original_text).text
            text = normalized_text
        else:
            text = original_text


        # Step 1: Analyze PII entities
        analyze_res = kwargs.pop("analyze_res", self._analyze(text, **kwargs))

        # Step 2: Anonymize with allow_list support
        anonymize_res, entity_mapping = self._anonymize(text, analyze_res, allow_list=allow_list, **kwargs)

        # Step 3: Replace numerical placeholders (if needed)
        replaced = self.replace_numerical_placeholders(anonymize_res.text)

        # Step 4: Return structured results
        return {
            "analysis": analyze_res,
            "anonymized": anonymize_res.text,
            "anonymized_items": anonymize_res.items,
            "entity_mapping": entity_mapping,
            "anonymized_with_placeholder": replaced,
        }
