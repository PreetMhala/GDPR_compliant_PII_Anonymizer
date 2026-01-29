import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.PIIDetector import PIIDetector

entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

# Prompt the user to enter the input text at runtime
text_input = input("Enter the query text to analyze: ")

detector_presidio = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": ["hr"],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": True,
            "ENABLE_CUSTOM_PATTERNS": True,
            "ENABLE_TRANSFORMERS": False,
        },
        "ENTITIES_TO_ALLOW": entities_to_allow
    }
)

print("PRESIDIO SPACY only results")
print("\n--------------------------------------\n")
print(detector_presidio.evaluate(text_input, language=["hr"]))


print("\n\n")
