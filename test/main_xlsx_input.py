import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.PIIDetector import PIIDetector
import ast


# Define the input and output file paths
input_file_path = 'test/croation_entity_specific_data_raw.xlsx'
output_file_path = 'test/outputs_croation_entity_specific_data_raw.xlsx'

# Load the input Excel file
df = pd.read_excel(input_file_path)

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

entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]

# Initialize the PIIDetector
detector = PIIDetector(config={
    "SUPPORTED_LANGUAGES": ["hr"],
    "PRESIDIO": {
        "ENABLE_SPACY": True,
        "ENABLE_FLAIR": True,
        "ENABLE_CUSTOM_PATTERNS": True,
        "ENABLE_TRANSFORMERS": False,
    }
})

# Prepare result containers
anonymized_texts = []
anonymized_mask_queries = []
anonymized_entities = []

# Wrap the DataFrame iterator with tqdm
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
    text_entry = row['Query']

    # Evaluate using PIIDetector
    result = detector.evaluate(text_entry, language="hr", ents=entities_to_anonymize, allow_list=entities_to_allow)

    anonymized_text = result['presidio'].get('anonymized', text_entry)
    anonymized_mask_query = result['presidio'].get('anonymized_with_placeholder', text_entry)
    entity_types = result['presidio'].get('anonymized_items', [])

    anonymized_texts.append(anonymized_text)
    anonymized_mask_queries.append(anonymized_mask_query)
    anonymized_entities.append(entity_types)

# Create output DataFrame
output_df = pd.DataFrame({
    'Entity': df['Entity'],
    'Query': df['Query'],
    'anonymized_text_after_reorganization': anonymized_texts,
    'anonymized_mask_query': anonymized_mask_queries,
    'anonymized_entities': anonymized_entities
})

# Save to Excel
output_df.to_excel(output_file_path, index=False)
print(f"\n Output saved to {output_file_path}")

# ---------------------Extract entity_type values ---------------------

# Re-load saved file to ensure compatibility
df_updated = pd.read_excel(output_file_path)

# Convert stringified list of dicts to actual list of dicts
def extract_entity_types(entity_list):
    try:
        # Handle actual list or stringified list
        entities = entity_list if isinstance(entity_list, list) else ast.literal_eval(str(entity_list))
        return list({ent.get('entity_type') for ent in entities if isinstance(ent, dict) and 'entity_type' in ent})
    except Exception:
        return []

df_updated['extracted_entity_types'] = df_updated['anonymized_entities'].apply(extract_entity_types)

# Save the updated DataFrame with extracted entities
df_updated.to_excel(output_file_path, index=False)
print(f"\n Updated file with 'extracted_entity_types' saved to {output_file_path}")
