import os

# Default language code (gets overwritten at runtime)
LANG_CODE = "pl"

# Base project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Static configs that don’t change
LANGDETECTOR_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "language_detector_config.yaml")

# ➤ This will be dynamically set in your main script at runtime (temp merged config)
ACTIVE_CONFIG_PATH = None

# ➤ Optional: If you still want access to the base language config paths, keep them:
ENGLISH_GERMAN_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_en_de_config_formatted.yaml")
CROATIAN_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_hr_config_formatted.yaml")
GREEK_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_el_config_formatted.yaml")
HUNGARIAN_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_hu_config_formatted.yaml")
POLISH_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_pl_config_formatted.yaml")
MONTENEGRIN_CONFIG_PATH = os.path.join(BASE_DIR, "yaml", "recognizer_cnr_config_formatted.yaml")

# ➤ But all your code should now use only ACTIVE_CONFIG_PATH

# Pattern Meghods
NAME = "name"
REGEX = "regex"
CONFIDENCE="confidence"
KEYWORDS="keywords"
PATTERNS="patterns"

NUMERICAL_PLACEHOLDER="numerical_placeholders"

TITLE="title"
TITLES_LIST="titles_list"
TITLE_PATTERN_REGEX="title_pattern_regex"
TITLE_PATTERNS ="title_patterns"
TITLE_CONTEXT="title_context_keyword"
PATTERNS_CONTEXT="pattern_context"

ZIP="zip"
ZIP_PATTERN="zip_pattern"
ZIP_CONTEXT="shared_keywords_zip_context"

PASSPORTS = "passports"
PASSPORT_LIST="passport_list"
PASSPORT_CONTEXT = "shared_keywords_passport_context"

ENTITY_COUNTER = "entity_counter"
ENTITY_TYPE= "entity_type"
ENTITY_MAPPING = "entity_mapping"

SPACY_RECOGNIZER = "SpacyRecognizer"
FLAIR_RECOGNIZER = "FlairRecognizer"

#custom pattern
PATTERN_CUSTOM="pattern_custom"
BILLING_NUMBER_RECOGNIZER ="billing_number_recognizer"
CUSTOM_BILLING_NUMBER="custom_billing_number"
SHARED_KEYWORD_BILLING= "shared_keywords_billing_number"

CREDIT_CARD_RECOGNIZER = "credit_card_recognizer"
CUSTOM_CREDIT_CARD = "custom_credit_card"
SHARED_KEYWORD_CREDITCARD ="shared_keywords_credit_card"

EMAIL_RECOGNIZER = "email_recognizer"
CUSTOM_EMAIL ="custom_email"
SHARED_KEYWORD_EMAIL ="shared_keywords_email"

IAN_RECOGNIZER ="ian_recognizer"
CUSTOM_IAN="custom_ian"
SHARED_KEYWORD_IAN = "shared_keywords_ian"

IBAN_RECOGNIZER ="iban_recognizer"
CUSTOM_IBAN= "custom_iban"
SHARED_KEYWORD_IBAN = "shared_keywords_iban"

ID_RECOGNIZER ="id_recognizer"
CUSTOM_ID="custom_id"
SHARED_KEYWORD_ID="shared_keywords_id"

IMEI_RECOGNIZER="imei_recognizer"
CUSTOM_IMEI= "custom_imei"
SHARED_KEYWORD_IMEI ="shared_keywords_imei"

IP_ADDR_RECOGNIZER ="ip_address_recognizer"
CUSTOM_IP_ADDR="custom_ip_address"
SHARED_KEYWORD_IP="shared_keywords_ip_address"

NETWORD_ADDRESS_RECOGNIZER ="network_address_recognizers"
CUSTOM_NETWORD_ADDRESS="custom_network_address"
SHARED_KEYWORD_NETWORK="shared_keywords_network_address"

PHONE_RECOGNIZER="phone_recognizer"
CUSTOM_PHONE="custom_phonenumber"
SHARED_KEYWORD_PHONENUMBER ="shared_keywords_phonenumber"

USSSN_RECOGNIZER ="usssn_recognizer" 
CUSTOM_USSSN ="custom_usssn"
SHARED_KEYWORD_USSSN ="shared_keywords_usssn"

URL_RECOGNIZER="url_recognizer"
CUSTOM_URL="custom_url"
SHARED_KEYWORD_URL="shared_keywords_url"

SSN_RECOGNIZER ="ssn_recognizer"
CUSTOM_SSN="custom_ssn"
SHARED_KEYWORD_SSN="shared_keywords_ssn"


URL_RECOG = "UrlRecognizer"
INPASSPORTRECOG="InPassportRecognizer"
ITPASSPORTRECOG="ItPassportRecognizer"
USPASSPORTRECOG="UsPassportRecognizer"
USSSNRECOG="UsSsnRecognizer"
CREDITCARDRECOG ="CreditCardRecognizer"
NLP_ENGINE="nlp_engine"


