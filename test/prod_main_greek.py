import sys
import os
import time
from gr_nlp_toolkit import Pipeline

# --- Import PIIDetector ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.PIIDetector import PIIDetector

# --- Setup Presidio Detector ---
entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]
detector_presidio = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": ["el"],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": True,
            "ENABLE_CUSTOM_PATTERNS": True,
            "ENABLE_TRANSFORMERS": False,
        },
        "ENTITIES_TO_ALLOW": entities_to_allow
    }
)

# --- Setup GR NLP Toolkit ---
nlp_gr = Pipeline("ner")

def run_gr_nlp_toolkit(text):
    doc = nlp_gr(text)
    tokens = doc.tokens
    reconstructed = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        ner_tag = token.ner
        word = token.text
        if ner_tag.startswith("B-"):
            entity_type = ner_tag[2:]
            i += 1
            while i < len(tokens) and tokens[i].ner.startswith("I-"):
                i += 1
            if i < len(tokens) and tokens[i].ner.startswith("E-"):
                i += 1
            reconstructed.append(f"<{entity_type}>")
        elif ner_tag.startswith("S-"):
            entity_type = ner_tag[2:]
            reconstructed.append(f"<{entity_type}>")
            i += 1
        else:
            reconstructed.append(word)
            i += 1
    return ' '.join(reconstructed)

# --- Loop for runtime input ---
print("\n⚡ Hybrid Anonymization (Presidio + GR NLP Toolkit)")
print("Type 'exit' to quit.\n")

while True:
    query = input("Enter Greek query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # Presidio Results
    print("\n--- Presidio Output ---")
    start = time.time()
    presidio_result = detector_presidio.evaluate(query, language=["el"])
    end = time.time()
    print(presidio_result)
    print(f"(Presidio Time: {end - start:.4f} seconds)")

    # GR NLP Toolkit Results
    print("\n--- GR NLP Toolkit Output ---")
    start = time.time()
    gr_result = run_gr_nlp_toolkit(query)
    end = time.time()
    print(gr_result)
    print(f"(GR NLP Toolkit Time: {end - start:.4f} seconds)")

    print("\n" + "-" * 50 + "\n")
