import sys
import os
import time
import importlib
import atexit
from tqdm import tqdm
import argparse
import pandas as pd
import ast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---- Set the desired language ----
lang_code = "el"

# ---- Step 1: Update the LANG_CODE in constants.py ----
from src.utils.language_code_updater import update_lang_code_in_constants
import src.utils.constants as constants_module

update_lang_code_in_constants(lang_code)
importlib.reload(constants_module)
LANG_CODE = constants_module.LANG_CODE

# ---- Step 2: Generate the temporary combined config ----
from src.utils.language_code_updater import generate_temp_combined_config

temp_config_path = generate_temp_combined_config(lang_code)
constants_module.ACTIVE_CONFIG_PATH = str(temp_config_path)

# ---- Step 3: Register cleanup ----
def cleanup_temp_file():
    if temp_config_path.exists():
        temp_config_path.unlink()
        print(f"[✓] Temporary config file deleted: {temp_config_path}")

atexit.register(cleanup_temp_file)

# ---- Step 4: Import PII Detector AFTER constants are set ----
from src.PIIDetector import PIIDetector

# ---- Entities to anonymize ----
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
    # Natco Specific
    "NATIONAL_ID_JMBG", "MONTENEGRIN_TAX_ID_PIB", "MONTENEGRIN_VISA", "LICENSE_PLATE_NUMBER_MNE", "DRIVER_LICENSE_NUMBER_MNE",
    "GREEK_PASSPORT", "GREECE_VISA", "GREECE_HEALTH_ID_NUMBER", "BOOKING_ACCOUNT", "CROATIAN_PASSPORT", "HUNGARIAN_PASSPORT",
    "POLAND_PASSPORT", "POLISH_TAX_ID_NIP", "POLISH_VISA", "PESEL", "LICENSE_PLATE_NUMBER"
]

entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

# ---- HARDCODED PATHS ----
input_file = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Greek_new_data_entity_specific_raw_with_validity.xlsx"
column_name = "Greek_Query"
output_file = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Greek_specific_outputs.xlsx"

# ---- Initialize Detector ----
detector = PIIDetector(
    config={
        "SUPPORTED_LANGUAGES": [lang_code],
        "PRESIDIO": {
            "ENABLE_SPACY": True,
            "ENABLE_FLAIR": True,
            "ENABLE_CUSTOM_PATTERNS": True
        },
        "ENTITIES_TO_ALLOW": entities_to_allow,
        "ENTITIES_TO_ANONYMIZE": entities_to_anonymize,
    }
)

# ---- Read Input ----
df = pd.read_excel(input_file)

# ---- Process each row ----
anonymized_queries = []
anonymized_placeholders = []
detected_entities = []

start_time = time.time()

for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing rows")):
    text = row[column_name]

    try:
        result = detector.evaluate(text, language=lang_code)

        presidio_data = result.get("presidio", {})
        anonymized_text = presidio_data.get("anonymized", text)
        anonymized_placeholder = presidio_data.get("anonymized_with_placeholder", text)
        entities = presidio_data.get("anonymized_items", [])

        anonymized_queries.append(anonymized_text)
        anonymized_placeholders.append(anonymized_placeholder)
        detected_entities.append(entities)

    except Exception as e:
        print(f"[Error] Row {i}: {str(e)}")
        anonymized_queries.append(text)
        anonymized_placeholders.append(text)
        detected_entities.append([])


end_time = time.time()

# ---- Add results to DataFrame ----
df["_Presidio_output"] = anonymized_queries
df["_Presidio_mask_query"] = anonymized_placeholders
df["_Presidio_entities"] = detected_entities

# ---- Save output before extracting entity types ----
df.to_excel(output_file, index=False)
print(f"✅ Output saved before entity type extraction to: {output_file}")

# ---- Read the same output file again ----
df = pd.read_excel(output_file)

# ---- Extract entity types ----
def extract_entity_types(entity_obj):
    try:
        if isinstance(entity_obj, str):
            entity_obj = ast.literal_eval(entity_obj)  # Convert from string if needed
        if isinstance(entity_obj, list):
            return [ent.get("entity_type") or ent.get("entity") for ent in entity_obj if isinstance(ent, dict)]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

df["_Presidio_entity_types"] = df["_Presidio_entities"].apply(extract_entity_types)

# ---- Save final output ----
df.to_excel(output_file, index=False)
print(f"✅ Final output saved to: {output_file}")
print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")
