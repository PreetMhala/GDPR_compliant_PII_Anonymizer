import sys
import os
import time
import importlib
import atexit

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---- Set the desired language ----
lang_code = "pl"

# ---- Step 1: Update the LANG_CODE in constants.py ----
from src.utils.language_code_updater import update_lang_code_in_constants
import src.utils.constants as constants_module

update_lang_code_in_constants(lang_code)
importlib.reload(constants_module)  # Force reload after modifying constants.py
LANG_CODE = constants_module.LANG_CODE

# ---- Step 2: Generate the temporary combined config ----
from src.utils.language_code_updater import generate_temp_combined_config

# Create temporary combined config
temp_config_path = generate_temp_combined_config(lang_code)

# Overwrite constants module dynamically
# constants_module.LANG_CODE = lang_code
constants_module.ACTIVE_CONFIG_PATH = str(temp_config_path)

# ---- Step 3: Register cleanup ----
def cleanup_temp_file():
    # Do not delete if lang_code is 'en' or 'de'
    if lang_code in ["en", "de"]:
        print(f"[✓] Reused config file for '{lang_code}', no deletion needed: {temp_config_path}")
        return

    # Delete only if the temp config file exists
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


# ---- Step 5: Run detection ----
entities_to_allow = ["Telekom", "Vodafone", "Magenta", "Prio", "Linkin Park", "d.velop", "d.ve"]
text_input = input("Enter the query text to analyze: ")

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

print(f"\n\n--- PII Detection Results for Language: {lang_code} ---\n")
start_time = time.time()
results = detector.evaluate(text_input, language=lang_code)
end_time = time.time()

for model_name, model_result in results.items():
    print(f"\n[{model_name.upper()} RESULTS]:\n")
    print(model_result)

print(f"\n\nExecution time: {end_time - start_time:.4f} seconds\n")
