import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.PIIDetector import PIIDetector
from src.log_utils import log_event
import ast
import uuid

# Generate a single traceId for this run
trace_id = str(uuid.uuid4())

# You could infer natco_code dynamically, e.g. from data or config, for now default:
default_usecase_version = "v1.0"
fallback_natco_code = "hu" #fallback if invalid or missing natco_code code is detected

# Log start of pipeline
log_event(
    step="Startup",
    event="Pipeline started",
    message="main.py execution started",
    natco_code="default_natco_code",
    usecase_version=default_usecase_version,
    traceId=trace_id
)

# Define the input and output file paths
input_file_path = '/Users/pritmhala/Downloads/PII_detector_Presidio_bertic_extended/Hungarian_model_selection_testing_data_Street_and_Person.xlsx'
output_file_path = '/Users/pritmhala/Downloads/PII_detector_Presidio_bertic_extended/Hungarian_model_selection_testing_data_Street_and_Person_results.xlsx'

# Load the input Excel file
df = pd.read_excel(input_file_path)

# Load input
try:
    df = pd.read_excel(input_file_path)
    log_event(
        step="FileRead",
        event="Input loaded",
        message=f"Loaded {len(df)} rows from Excel",
        natco_code="default_natco_code",
        usecase_version=default_usecase_version,
        traceId=trace_id
    )
except Exception as e:
    log_event(
        step="FileRead",
        event="Failed to read Excel",
        message=str(e),
        natco_code="default_natco_code",
        usecase_version=default_usecase_version,
        log_level="Error",
        traceId=trace_id
    )
    raise e


# Define entities to anonymize and whitelist settings
entities_to_anonymize = [
    "IN_VOTER", "IN_PAN", "DRIVER_LICENSE", "SWIFT_CODE", "AU_ACN", "NETWORK_ADDRESS", "LICENSE_PLATE", "ID",
    "COORDINATE", "ZIP_CODE", "CREDIT_CARD", "NRP", "BILLING_NR", "ORGANIZATION", "SG_NRIC_FIN", "IBAN", "CRYPTO",
    "GERMAN_ID", "AU_MEDICARE", "EMAIL_ADDRESS", "GPE", "CURRENCY", "SSN", "AU_TFN", "IMEI_NUMBER", "IN_VEHICLE_REGISTRATION",
    "AUSTRIAN_CUSTOMER_NUMBER", "IBAN_CODE", "PHONE_NUMBER", "PASSPORT", "BIG_NUMBERS", "GERMAN_VISA", "CARD_PROFILE_NR", "STATE",
    "IN_AADHAAR", "TITLE", "STREET_ADDRESS", "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "AU_ABN", "IMEI", "STREET", "DATE",
    "IP_ADDRESS", "LOCATION", "PERSON", "ROMANIAN_ID", "PII", "UK_NHS", "US_BANK_NUMBER", "URL", "TAX_IDENTIFICATION_NR",
    "SOCIAL_MEDIA_HANDLE", "PASSWORD", "CONTRACT_ENTR_NR", "ROUTING_NUMBER", "CUSTOMER_NR", "GERMAN_PRE_ID", "GPS_COORDINATES",
    "US_ITIN", "AGE", "POLISH_ID", "INTERNET_ACCESS_NR", "MAC_ADDRESS", "INDIAN_PASSPORT", "US_PASSPORT", "ITALY_PASSPORT",
    "CANADA_PASSPORT", "FRANCE_PASSPORT", "GERMAN_PASSPORT", "SWEDEN_PASSPORT", "UK_PASSPORT", "AUSTRIA_PASSPORT",
    "RUSSIAN_PASSPORT", "GERMAN_SSN", "OIB", "CROATIAN_VISA"
]

entities_to_anonymize = [
    # Original list first
    "IN_VOTER", "IN_PAN", "DRIVER_LICENSE", "SWIFT_CODE", "AU_ACN", "NETWORK_ADDRESS", "LICENSE_PLATE", "ID",
    "COORDINATE", "ZIP_CODE", "CREDIT_CARD", "NRP", "BILLING_NR", "ORGANIZATION", "SG_NRIC_FIN", "IBAN", "CRYPTO",
    "GERMAN_ID", "AU_MEDICARE", "EMAIL_ADDRESS", "GPE", "CURRENCY", "SSN", "AU_TFN", "IMEI_NUMBER", "IN_VEHICLE_REGISTRATION",
    "AUSTRIAN_CUSTOMER_NUMBER", "IBAN_CODE", "PHONE_NUMBER", "PASSPORT", "BIG_NUMBERS", "GERMAN_VISA", "CARD_PROFILE_NR", "STATE",
    "IN_AADHAAR", "TITLE", "STREET_ADDRESS", "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "AU_ABN", "IMEI", "STREET", "DATE",
    "IP_ADDRESS", "LOCATION", "PERSON", "ROMANIAN_ID", "PII", "UK_NHS", "US_BANK_NUMBER", "URL", "TAX_IDENTIFICATION_NR",
    "SOCIAL_MEDIA_HANDLE", "PASSWORD", "CONTRACT_ENTR_NR", "ROUTING_NUMBER", "CUSTOMER_NR", "GERMAN_PRE_ID", "GPS_COORDINATES",
    "US_ITIN", "AGE", "POLISH_ID", "INTERNET_ACCESS_NR", "MAC_ADDRESS", "INDIAN_PASSPORT", "US_PASSPORT", "ITALY_PASSPORT",
    "CANADA_PASSPORT", "FRANCE_PASSPORT", "GERMAN_PASSPORT", "SWEDEN_PASSPORT", "UK_PASSPORT", "AUSTRIA_PASSPORT",
    "RUSSIAN_PASSPORT", "GERMAN_SSN", "OIB", "CROATIAN_VISA",

    # Extra placeholders not present above
    "NATIONAL_ID_JMBG", "MONTENEGRIN_TAX_ID_PIB", "MONTENEGRIN_VISA", "LICENSE_PLATE_NUMBER_MNE", "DRIVER_LICENSE_NUMBER_MNE",
    "GREEK_PASSPORT", "GREECE_VISA", "GREECE_HEALTH_ID_NUMBER", "BOOKING_ACCOUNT", "CROATIAN_PASSPORT", "HUNGARIAN_PASSPORT",
    "POLAND_PASSPORT", "POLISH_TAX_ID_NIP", "POLISH_VISA", "PESEL", "LICENSE_PLATE_NUMBER"
]


entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

# natco specific config calling

# Initialize the Hybrid Detector
detector = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": ["hu"],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": False,
            "ENABLE_CUSTOM_PATTERNS": True,
            "ENABLE_TRANSFORMERS": False,
            "ENABLE_BERTIC": False
        },
        "ENTITIES_TO_ALLOW": entities_to_allow
    }
)

log_event(
    step="Setup",
    event="PIIDetector initialized",
    message="Anonymization engine configured",
    natco_code="default_natco_code",
    usecase_version=default_usecase_version,
    model="Presidio+BERTIC",
    traceId=trace_id
)

# Output lists
presidio_outputs, presidio_mask_queries, presidio_entities = [], [], []

log_event(
    step="Anonymization",
    event="Processing started",
    message=f"Running Presidio and BERTIC on {len(df)} queries",
    natco_code="default_natco_code",
    usecase_version=default_usecase_version,
    traceId=trace_id
)

import time

start_time = time.time()

# Run detection and store raw outputs
for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Running Presidio and BERTIC")):
    text_entry = row['Hungarian_Query']
    natco_code = row.get('natco_code') or fallback_natco_code
    try:
        result = detector.evaluate(text_entry, language=natco_code.lower())

        # --- Presidio ---
        presidio_data = result.get("presidio", {})
        anonymized_presidio = presidio_data.get("anonymized", text_entry)
        anonymized_placeholder = presidio_data.get("anonymized_with_placeholder", text_entry)
        presidio_items = presidio_data.get("anonymized_items", [])

        presidio_outputs.append(anonymized_presidio)
        presidio_mask_queries.append(anonymized_placeholder)
        presidio_entities.append(presidio_items)

        # Log success
        log_event(
            step="Anonymization",
            event="Row processed",
            message=f"Row {i} processed successfully",
            natco_code=natco_code,
            usecase_version=default_usecase_version,
            model="Presidio+BERTIC",
            traceId=trace_id
        )

    except Exception as e:
        log_event(
            step="Anonymization",
            event="Processing failed",
            message=f"Row {i}: {str(e)}",
            natco_code=natco_code,
            usecase_version=default_usecase_version,
            model="Presidio+BERTIC",
            log_level="Error",
            traceId=trace_id
        )
        presidio_outputs.append(text_entry)
        presidio_mask_queries.append(text_entry)
        presidio_entities.append([])

log_event(
    step="Anonymization",
    event="Raw outputs complete",
    message="Presidio and BERTIC anonymization complete. Preparing to save outputs.",
    natco_code=natco_code,
    usecase_version=default_usecase_version,
    traceId=trace_id
)

# Add all new columns
df['_Presidio_output'] = presidio_outputs
df['_Presidio_mask_query'] = presidio_mask_queries
df['_Presidio_entities'] = presidio_entities

# Save to Excel
df.to_excel(output_file_path, index=False)
log_event(
    step="SaveOutput",
    event="Initial output saved",
    message=f"Excel file saved to {output_file_path} (Presidio, BERTIC, Hybrid outputs)",
    natco_code=natco_code,
    usecase_version=default_usecase_version,
    traceId=trace_id
)

# Re-load for post-processing
df = pd.read_excel(output_file_path)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

# Entity type extractor
def extract_entity_types(entity_obj):
    try:
        if isinstance(entity_obj, str):
            entity_obj = ast.literal_eval(entity_obj)
        if isinstance(entity_obj, list) and len(entity_obj) > 0 and isinstance(entity_obj[0], list):
            entity_obj = entity_obj[0]
        return [ent.get('entity_type') or ent.get('entity') for ent in entity_obj if isinstance(ent, dict)]
    except (ValueError, SyntaxError, TypeError):
        return []

# Extract types
df['_Presidio_entity_types'] = df['_Presidio_entities'].apply(extract_entity_types)

# Final save
df.to_excel(output_file_path, index=False)
log_event(
    step="SaveOutput",
    event="Final output saved",
    message=f"Entity types extracted and final Excel saved to {output_file_path}",
    natco_code=natco_code,
    usecase_version=default_usecase_version,
    traceId=trace_id
)

print(f"✅ Processing complete. Output saved to: {output_file_path}")
