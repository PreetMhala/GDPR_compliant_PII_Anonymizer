import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.PIIDetector import PIIDetector

# Entities to allow through during evaluation
entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

# Prompt for input text
text_input = input("Enter the query text to analyze: ")


# Initialize the PIIDetector with both Presidio and BERTIC enabled
detector = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": ["en","de"],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": True,
            "ENABLE_CUSTOM_PATTERNS": True,
            "ENABLE_TRANSFORMERS": True,
            "ENABLE_BERTIC": True
        },
        "ENTITIES_TO_ALLOW": entities_to_allow
    }
)

# Evaluate using available models
print("\n\n--- PII Detection Results ---\n")
start_time = time.time()
results = detector.evaluate(text_input, language="en")
end_time = time.time()

# Display results for each model
for model_name, model_result in results.items():
    print(f"\n[{model_name.upper()} RESULTS]:\n")
    print(model_result)

print(f"\n\nExecution time: {end_time - start_time:.4f} seconds\n")
