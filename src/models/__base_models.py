import datetime
from abc import ABC
from transformers import pipeline
import re
import logging

logger = logging.getLogger("pii-identifier")


class _TransformersNERBaseModel(ABC):
    # task = "ner"
    #
    def __init__(self, task: str, model_name: str, tokenizer_kwargs: dict = None,
                 pipeline_kwargs: dict = None, **kwargs):
        # super().__init__(**{k: v for k, v in kwargs.items() if k != 'task'})  # Exclude 'task'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=True)
        pipeline_kwargs = pipeline_kwargs or {}
        if tokenizer_kwargs is not None:
            pipeline_kwargs["tokenizer"] = (model_name, tokenizer_kwargs)
        self.pipe = pipeline(
            task,
            model=model_name,
            **pipeline_kwargs,
        )
        self.NER_LABELS = {"entity": "entity", "end": "end", "start": "start", "score": "score",
                           "word": "word", **kwargs.pop("NER_LABELS", {})}
        self.model_name = model_name
        logger.info(f"Loaded NER model: {model_name}")

    def _analyze(self, text: str, ents: list = None, thres: float = 0.35, **kwargs):
        def rename_keys(d: dict) -> dict:
            d["entity"] = d.pop(self.NER_LABELS['entity'])
            d["score"] = d.pop(self.NER_LABELS['score'])
            d["word"] = d.pop(self.NER_LABELS['word'])
            d["start"] = d.pop(self.NER_LABELS['start'])
            d["end"] = d.pop(self.NER_LABELS['end'])
            return d

        d_start_time = datetime.datetime.now()
        results = [rename_keys(e) for e in self.pipe(text)]
        d_time = datetime.datetime.now() - d_start_time
        logger.debug("-- [%s] evaluated [%s]: %s.%s seconds", self.model_name.upper(), d_time.seconds, d_time.microseconds)
        return [e for e in results if (ents is None or e["entity"] in ents) and e["score"] > thres]

    def _anonymize(self, text: str, analyze_res=None, **kwargs):
        analyze_res = analyze_res if analyze_res is not None else self._analyze(text, **kwargs)
        for i in sorted(analyze_res, key=lambda x: (x['start'], x['end']), reverse=True):
            # print(i)
            text = f"{text[:i['start']]}{' ' * (i['start'] != 0)}<{i['entity']}>{text[i['end']:]}"
        return re.sub(r"<(?P<entity>.*?)><(?P=entity)>", r"<\1>", text)

    def denonymize(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, text: str, **kwargs):
        d_start_time = datetime.datetime.now()
        analyze_res = kwargs.pop("analyze_res", self._analyze(text, **kwargs))
        ret = {"analysis": analyze_res, "anonymized": self._anonymize(text, analyze_res, **kwargs)}
        d_time = datetime.datetime.now() - d_start_time
        logger.debug("-- [%s] evaluated [%s]: %s.%s seconds", self.model_name.upper(), d_time.seconds,
                     d_time.microseconds)
        return ret


class _TransformersMaskBaseModel(ABC):
    def __init__(self, model_name: str, tokenizer_kwargs: dict = None, pipeline_kwargs: dict = None, **kwargs):
        # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=True)
        pipeline_kwargs = pipeline_kwargs or {}
        if tokenizer_kwargs is not None and isinstance(tokenizer_kwargs, dict):
            pipeline_kwargs["tokenizer"] = (model_name, tokenizer_kwargs)
        self.pipe = pipeline(
            task="text2text-generation",
            model=model_name,
            device_map="auto",
            **pipeline_kwargs,
        )
        self.mask_regex = re.compile(kwargs.pop("mask_regex", r"<(.+?)(?:_\d)?>"))
        self.model_name = model_name
        logger.info(f"Loaded Mask model: {model_name}")

    def _analyze(self, text: str, ents: list = None, _anonymize_res: str = None, **kwargs):
        _anonymize_res = _anonymize_res if _anonymize_res is not None else self._anonymize(text, **kwargs)
        regex_build = self.mask_regex.sub(lambda m: rf"(?P<{m.group(1)}_{m.start(1)}>.+?)", re.escape(_anonymize_res))
        logger.debug("-- [%s] regex_build: '%s'", self.model_name.upper(), regex_build)
        match = re.fullmatch(regex_build, text)
        if match is not None:
            d_start_time = datetime.datetime.now()
            ret = [e for e in [
                {"entity": re.sub(r"_\d+$", "", ent), "word": word, "start": match.start(ent),
                 "end": match.end(ent), "score": 0} for ent, word in match.groupdict().items()] if
                   (ents is None or e["entity"] in ents)]
            d_time = datetime.datetime.now() - d_start_time
            logger.debug("-- [%s] analyzed [%s]: %s.%s seconds", self.model_name.upper(), d_time.seconds,
                         d_time.microseconds)
            return ret

    def _anonymize(self, text: str, **kwargs):
        anonymized: dict = self.pipe(text,
                                     max_length=self.pipe.tokenizer.model_max_length)  # len(self.pipe.tokenizer.tokenize(text)) * 2
        answer = [*anonymized[0].values()][0]
        return re.sub(r"<(?P<entity>.*?)><(?P=entity)>", r"<\1>", re.sub(r"\[([A-Z0-9_]+?)]", r"<\1>", answer))

    def denonymize(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, text: str, **kwargs):
        """
        Evaluates PII using mask based transformers models.
        :param text: Text to evaluate.
        :keyword anonymize_res: Anonymized text.
        :return:
        """
        try:
            d_start_time = datetime.datetime.now()
            anonymize_res = kwargs.pop("anonymize_res", self._anonymize(text, **kwargs))
            ret = {"analysis": self._analyze(text, _anonymize_res=anonymize_res, **kwargs), "anonymized": anonymize_res}
            d_time = datetime.datetime.now() - d_start_time
            logger.debug("-- [%s] evaluated [%s]: %s.%s seconds", self.model_name.upper(), d_time.seconds,
                         d_time.microseconds)
            return ret
        except Exception as e:
            logger.error(f"Error evaluating '%s' using mask based transformers model [%s]\nReturning None", text,
                         self.model_name)
            logger.exception(e)
            return {"analysis": None, "anonymized": None}
