from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_checkpoint = "pczarnik/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Czy możesz potwierdzić, czy Cezary Gajda jest wykładowcą na Wydziale Informatyki Politechniki Łódzkiej?"

ner_results = nlp(example)
print(ner_results)
