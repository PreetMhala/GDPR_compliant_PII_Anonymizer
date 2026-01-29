import os
import sys
import uuid
import re
import time
import ast
import pandas as pd
from tqdm import tqdm
from gr_nlp_toolkit import Pipeline
from src.PIIDetector import PIIDetector
from src.log_utils import log_event

# --- Configuration ---
input_file_path = '/Users/pritmhala/Downloads/PII_Detector_base_ver2 2/Greek_new_data_entity_specific_raw_with_validity.xlsx'
output_file_path = '/Users/pritmhala/Downloads/PII_Detector_base_ver2 2/Greek_new_data_entity_specific_raw_with_validity_Outputs_FILE.xlsx'

default_usecase_version = "v1.0"
trace_id = str(uuid.uuid4())

log_event("Startup", "Pipeline started", "Batch Presidio + GR NLP execution", "NA", default_usecase_version, traceId=trace_id)

# --- Load Input ---
try:
    df = pd.read_excel(input_file_path)
    log_event("FileRead", "Input loaded", f"Loaded {len(df)} rows from Excel", "NA", default_usecase_version, traceId=trace_id)
except Exception as e:
    log_event("FileRead", "Failed to read Excel", str(e), "NA", default_usecase_version, log_level="Error", traceId=trace_id)
    raise e

# --- Initialize Detectors ---
entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

detector = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": ["el"],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": True,
            "ENABLE_CUSTOM_PATTERNS": True,
            "ENABLE_TRANSFORMERS": False,
        },
        "ENTITIES_TO_ALLOW": entities_to_allow
    },
    traceId=trace_id,
    natco_code="NA",
    usecase_version=default_usecase_version
)

nlp_gr = Pipeline("ner")

def run_gr_nlp_toolkit(text):
    try:
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
                reconstructed.append(f"<{ner_tag[2:]}>")
                i += 1
            else:
                reconstructed.append(word)
                i += 1
        return ' '.join(reconstructed)
    except Exception as e:
        return f"[GR-NLP ERROR: {str(e)}]"

# --- Output Lists ---
presidio_outputs, presidio_mask_queries, presidio_entities = [], [], []
gr_outputs = []

log_event("Anonymization", "Processing started", f"Running all engines on {len(df)} queries", "NA", default_usecase_version, traceId=trace_id)

# --- Main Loop ---
for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Presidio + GR_NLP_TOOLKIT Anonymization")):
    text_entry = row.get('Greek_Query')

    try:
        result = detector.evaluate(text_entry, language="el")

        # Presidio
        presidio_data = result.get("presidio", {})
        presidio_outputs.append(presidio_data.get("anonymized", text_entry))
        presidio_mask_queries.append(presidio_data.get("anonymized_with_placeholder", text_entry))
        presidio_entities.append(presidio_data.get("anonymized_items", []))

        # GR NLP
        gr_outputs.append(run_gr_nlp_toolkit(text_entry))

        log_event("Anonymization", "Row processed", f"Row {i} processed", "NA", default_usecase_version, model="Presidio+GR", traceId=trace_id)

    except Exception as e:
        log_event("Anonymization", "Error", f"Row {i}: {str(e)}", "NA", default_usecase_version, log_level="Error", traceId=trace_id)
        presidio_outputs.append(text_entry)
        presidio_mask_queries.append(text_entry)
        presidio_entities.append([])
        gr_outputs.append(f"[GR-NLP ERROR: {str(e)}]")

# --- Save Columns ---
df['_Presidio_output'] = presidio_outputs
df['_Presidio_mask_query'] = presidio_mask_queries
df['_Presidio_entities'] = presidio_entities
df['_GR_NLP_output'] = gr_outputs

# Re-load for post-processing
df = pd.read_excel(output_file_path)

# --- Entity Types Extraction ---
def extract_entity_types(entity_obj):
    try:
        if isinstance(entity_obj, str):
            entity_obj = ast.literal_eval(entity_obj)
        if isinstance(entity_obj, list) and len(entity_obj) > 0 and isinstance(entity_obj[0], list):
            entity_obj = entity_obj[0]
        return [ent.get('entity_type') or ent.get('entity') for ent in entity_obj if isinstance(ent, dict)]
    except (ValueError, SyntaxError, TypeError):
        return []

def extract_gr_nlp_entities(text):
    try:
        return re.findall(r'<(.*?)>', text)
    except Exception:
        return []

df['_Presidio_entity_types'] = df['_Presidio_entities'].apply(extract_entity_types)
df['_GR_NLP_output_types'] = df['_GR_NLP_output'].apply(extract_gr_nlp_entities)

df.to_excel(output_file_path, index=False)
log_event("SaveOutput", "Saved to Excel", f"Final output written to {output_file_path}", "NA", default_usecase_version, traceId=trace_id)

print(f"✅ Done! Output saved to:\n{output_file_path}")
