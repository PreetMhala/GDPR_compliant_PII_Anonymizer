import copy
import datetime
import logging
from typing import Optional, List, Tuple, Set, Union
from src.utils.constants import LANG_CODE

from presidio_analyzer import RecognizerResult, EntityRecognizer, AnalysisExplanation
from presidio_analyzer.nlp_engine import NlpArtifacts

from src.models.__base_models import _TransformersNERBaseModel, _TransformersMaskBaseModel

# from .configuration import BERT_DEID_CONFIGURATION
BERT_DEID_CONFIGURATION = {}

logger = logging.getLogger("pii-identifier")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, TokenClassificationPipeline
except ImportError:
    raise ImportError("`transformers` is not installed.")


class _TransformersRec(EntityRecognizer):
    """
    A configuration object should be maintained for each dataset-model combination and translate
    entities names into a standardized view. A sample of a configuration file is attached in
    the example.
    :param supported_entities: List of entities to run inference on
    :type supported_entities: Optional[List[str]]
    """
    DEFAULT_EXPLANATION = "Identified as {} by the {name} NER model"
    ENTITIES = ["LOCATION", "PREFIX", "PERSON", "ORGANIZATION", "AGE", "PHONE_NUMBER", "EMAIL", "DATE_TIME", "DEVICE",
                "PROFESSION", "USERNAME", "ID", "IP_ADDRESS", "MAC_ADDRESS", "PASSWORD", "CREDIT_CARD", "ID_NUM", "ZIP",
                "CREDIT_CARD_CVV", "US_SSN", "PHONE_IMEI", "CRYPTO", "GENDER", "CURRENCY", "TAX_NUMBER", "ACCOUNT_NUM",
                "DRIVER_LICENSE_NUM", "CITY", "STATE", "COUNTY", "STREET_ADDRESS", "ZIP_CODE", "COORDINATE"]
    LABELS_TO_IGNORE = [
        "O", "URL", "PIN", "USERNAME", "HEIGHT", "AMOUNT", "JOBTYPE", "EYECOLOR",
        "ORDINALDIRECTION", "MASKEDNUMBER", "USERAGENT"
    ]
    MODEL_TO_PRESIDIO_MAPPING = {
        "PREFIX": "PREFIX",
        "PER": "PERSON",
        "PERSON": "PERSON",
        "PATIENT": "PERSON",
        "GIVENNAME": "PERSON",
        "SURNAME": "PERSON",
        "STAFF": "PERSON",
        "HCW": "PERSON",
        "FIRSTNAME": "PERSON",
        "MIDDLENAME": "PERSON",
        "LASTNAME": "PERSON",
        "ACCOUNTNAME": "PERSON",
        "SEX": "GENDER",
        "CURRENCY": "CURRENCY",
        "CURRENCYNAME": "CURRENCY",
        "CURRENCYCODE": "CURRENCY",
        "CURRENCYSYMBOL": "CURRENCY",
        "GENDER": "GENDER",
        "AGE": "AGE",
        "ID": "ID",
        "BIC": "ID",
        "IBAN": "ID",
        "VEHICLEVRM": "ID",
        "VEHICLEVIN": "ID",
        "IDCARDNUM": "ID",
        "LOC": "LOCATION",
        "CITY": "CITY",
        "STATE": "STATE",
        "COUNTY": "COUNTY",
        "STREET": "STREET_ADDRESS",
        "ZIPCODE": "ZIP_CODE",
        "JOBAREA": "LOCATION",
        "SECONDARYADDRESS": "LOCATION",
        "BUILDINGNUMBER": "LOCATION",
        "NEARBYGPSCOORDINATE": "COORDINATE",
        "HOSP": "LOCATION",
        "HOSPITAL": "LOCATION",
        "BUILDINGNUM": "LOCATION",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "DOB": "DATE_TIME",
        "DATEOFBIRTH": "DATE_TIME",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE_NUMBER",
        "PHONENUMBER": "PHONE_NUMBER",
        "ORG": "ORGANIZATION",
        "PATORG": "ORGANIZATION",
        "VENDOR": "ORGANIZATION",
        "COMPANYNAME": "ORGANIZATION",
        "IP": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPV6": "IP_ADDRESS",
        "MAC": "MAC_ADDRESS",
        "BITCOINADDRESS": "CRYPTO",
        "LITECOINADDRESS": "CRYPTO",
        "ETHEREUMADDRESS": "CRYPTO",
        "SOCIALNUM": "US_SSN",
        "SSN": "US_SSN",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "CREDITCARDISSUER": "ORGANIZATION",
        "CREDITCARDCVV": "CREDIT_CARD_CVV",
        "PHONEIMEI": "PHONE_IMEI",
        "PASSWORD": "PASSWORD",
        "JOBTITLE": "PROFESSION",
        "ACCOUNTNUMBER": "ACCOUNT_NUM",
        "ACCOUNTNUM": "ACCOUNT_NUM",
        "DRIVERLICENSENUM": "DRIVER_LICENSE_NUM",
        "TAXNUM": "TAX_NUMBER",
        "TELEPHONENUM": "PHONE_NUMBER",
    }

    def load(self) -> None:
        pass

    def __init__(
            self,
            # model_path: Optional[str] = None,
            name: Optional[str] = None,
            supported_language: str = "en, hr",
            supported_entities: Optional[List[str]] = None,
            check_label_groups: Optional[Tuple[Set, Set]] = None,
            model: Union[_TransformersNERBaseModel, _TransformersMaskBaseModel, None] = None,
            sub_word_aggregation: str = "simple",
            overlap_length: int = 40,
            chunk_length: int = 600,
            id_score_reduction: float = 0.4,
            id_entity_name: str = "ID",
            max_length: Optional[int] = None,
    ):
        self.check_label_groups = check_label_groups  # or self.CHECK_LABEL_GROUPS
        supported_entities = supported_entities or self.ENTITIES
        logger.info(f"[PRESIDIO_TRANSFORMERS] Loading Transformers model - {model.__class__.__name__}")
        self.model = model

        super().__init__(supported_entities, name, supported_language)

        self.aggregation_mechanism = sub_word_aggregation
        if chunk_length <= overlap_length:
            logger.warning(
                "overlap_length should be shorter than chunk_length, setting overlap_length to by half of chunk_length")
            overlap_length = chunk_length // 2
        self.overlap_length = overlap_length
        self.chunk_length = chunk_length
        self.id_score_reduction = id_score_reduction
        self.id_entity_name = id_entity_name
        self.model_to_presidio_mapping = self.MODEL_TO_PRESIDIO_MAPPING
        self.ignore_labels = self.LABELS_TO_IGNORE
        self.max_length = max_length or self.model.pipe.tokenizer.model_max_length

    # Class to use transformers with Presidio as an external recognizer.
    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None) -> List[RecognizerResult]:
        """
        Analyze text using transformers model to produce NER tagging.
        :param text: The text for analysis.
        :param entities: The list of entities this recognizer is able to detect.
        :param nlp_artifacts: Not used by this recognizer.
        :return: The list of Presidio RecognizerResult constructed from the recognized transformers detections.
        """
        results = list()
        # Run transformer model on the provided text
        ner_results = self._get_ner_results_for_text(text)

        a_start_time = datetime.datetime.now()
        for res in ner_results:
            res["entity"] = self.__check_label_transformer(res["entity"])
            if not res["entity"] or res["entity"] not in entities:
                continue

            if res["entity"] == self.id_entity_name:
                # print(f"ID entity found, multiplying score by {self.id_score_reduction}")
                res["score"] = res["score"] * self.id_score_reduction

            text_exp = self.DEFAULT_EXPLANATION.format(res["entity"], name=self.name)
            explanation = self.build_transformers_explanation(float(round(res["score"], 2)), text_exp, res["word"])
            transformers_result = self._convert_to_recognizer_result(res, explanation)

            results.append(transformers_result)

        a_time = datetime.datetime.now() - a_start_time
        logger.debug("-- [PRESIDIO_TRANSFORMERS] evaluated_transformers_%s: %s.%s seconds", self.supported_language, a_time.seconds, a_time.microseconds)
        return results

    @staticmethod
    def split_text_to_word_chunks(
            input_length: int, chunk_length: int, overlap_length: int
    ) -> List[List]:
        """The function calculates chunks of text with size chunk_length. Each chunk has overlap_length number of
        words to create context and continuity for the model

        :param input_length: Length of input_ids for a given text
        :type input_length: int
        :param chunk_length: Length of each chunk of input_ids.
        Should match the max input length of the transformer model
        :type chunk_length: int
        :param overlap_length: Number of overlapping words in each chunk
        :type overlap_length: int
        :return: List of start and end positions for individual text chunks
        :rtype: List[List]
        """
        if input_length < chunk_length:
            return [[0, input_length]]
        return [
            [i, min([i + chunk_length, input_length])]
            for i in range(
                0, input_length - overlap_length, chunk_length - overlap_length
            )
        ]

    def _get_ner_results_for_text(self, text: str) -> List[dict]:
        """The function runs model inference on the provided text.
        The text is split into chunks with n overlapping characters.
        The results are then aggregated and duplications are removed.

        :param text: The text to run inference on
        :type text: str
        :return: List of entity predictions on the word level
        :rtype: List[dict]
        """
        # calculate inputs based on the text
        text_length = len(text)
        # split text into chunks
        if text_length <= self.max_length:
            predictions = self.model._analyze(text)
        else:
            logger.info(
                f"splitting the text into chunks, length {text_length} > {self.max_length}"
            )
            predictions = list()
            chunk_indexes = _TransformersRec.split_text_to_word_chunks(text_length, self.chunk_length,
                                                                       self.overlap_length)

            # iterate over text chunks and run inference
            for chunk_start, chunk_end in chunk_indexes:
                chunk_text = text[chunk_start:chunk_end]
                chunk_preds = self.model._analyze(chunk_text)

                # align indexes to match the original text - add to each position the value of chunk_start
                aligned_predictions = list()
                for prediction in chunk_preds:
                    prediction_tmp = copy.deepcopy(prediction)
                    prediction_tmp[self.model.NER_LABELS.get("start")] += chunk_start
                    prediction_tmp[self.model.NER_LABELS.get("end")] += chunk_start
                    aligned_predictions.append(prediction_tmp)

                predictions.extend(aligned_predictions)

        # remove duplicates
        predictions = [dict(t) for t in {tuple(d.items()) for d in predictions}]
        return predictions

    @staticmethod
    def _convert_to_recognizer_result(
            prediction_result: dict, explanation: AnalysisExplanation
    ) -> RecognizerResult:
        """The method parses NER model predictions into a RecognizerResult format to enable down the stream analysis

        :param prediction_result: A single example of entity prediction
        :type prediction_result: dict
        :param explanation: Textual representation of model prediction
        :type explanation: str
        :return: An instance of RecognizerResult which is used to model evaluation calculations
        :rtype: RecognizerResult
        """

        transformers_results = RecognizerResult(
            entity_type=prediction_result["entity"],
            start=prediction_result["start"],
            end=prediction_result["end"],
            score=float(round(prediction_result["score"], 2)),
            analysis_explanation=explanation,
        )

        return transformers_results

    def build_transformers_explanation(
            self,
            original_score: float,
            explanation: str,
            pattern: str,
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :param pattern: Regex pattern used
        :return Structured explanation and scores of a NER model prediction
        :rtype: AnalysisExplanation
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=float(original_score),
            textual_explanation=explanation,
            pattern=pattern,
        )
        return explanation

    def __check_label_transformer(self, label: str) -> Optional[str]:
        """The function validates the predicted label is identified by Presidio
        and maps the string into a Presidio representation
        :param label: Predicted label by the model
        :return: Returns the adjusted entity name
        """

        # convert model label to presidio label
        entity = self.model_to_presidio_mapping.get(label, None)

        if entity in self.ignore_labels:
            return None

        if entity is None:
            logger.warning(f"Found unrecognized label {label}, returning entity as is")
            return label

        if entity not in self.supported_entities:
            logger.warning(f"Found entity {entity} which is not supported by Presidio")
            return entity
        return entity
