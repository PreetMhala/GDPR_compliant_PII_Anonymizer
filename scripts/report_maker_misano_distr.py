import pandas as pd
from collections import Counter

# Sandbox folder
# Define file paths
input_file = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Montenegrin_specific_outputs.xlsx"
output_file = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Report_Montenegrin_specific_outputs.xlsx"

entities = [
    "IP_ADDRESS",
    "IBAN",
    "CUSTOMER_ACCOUNT",
    "CUSTOMER",
    "GPS_COORDINATES",
    "BILLING_NR",
    "CREDIT_CARD",
    "TAX_IDENTIFICATION_NUMBER",
    "BOOKING_ACCOUNT",
    "CARD_PROFILE_NR",
    "URL",
    "MAC_ADDRESS",
    "IMEI",
    "GERMAN_VISA",
    "EMAIL_ADDRESS",
    "SSN",
    "STREET",
    "PHONE_NUMBER",
    "INTERNET_ACCESS_NUMBER",
    "PASSPORT",
    "PERSON",
    "SOCIAL_MEDIA_HANDLE",
    "CROATIAN_VISA",
    "DRIVER_LICENSE",
    "DATE",
    "ZIP_CODE",
    "OIB",
]




# Load the input Excel file
df = pd.read_excel(input_file)

# Initialize result dictionary
results = {entity: [0] * 26 for entity in entities}  # Extended for 4 new columns

# Helper function to compute mis-anonymisation distribution
def get_mis_anonymisation_distribution(sub_df):
    counter = Counter()
    for _, row in sub_df.iterrows():
        presidio_types = eval(row['_Presidio_entity_types']) if pd.notna(row['_Presidio_entity_types']) else []
        if not presidio_types:
            # If empty, check gr nlp types
            gr_nlp_type = eval(row['_BERTIC_entity_types']) if pd.notna(row['_BERTIC_entity_types']) else []
            counter.update(gr_nlp_type)
        else:
            counter.update(presidio_types)
    return dict(counter)

# Process each entity
for entity in entities:
    valid_df = df[(df["Entity"] == entity) & (df["Validity"] == "Valid")]
    invalid_df = df[(df["Entity"] == entity) & (df["Validity"] == "Invalid")]
    valid_cases = valid_df.shape[0]
    invalid_cases = invalid_df.shape[0]

    # Base counts
    results[entity][0] = valid_cases
    results[entity][1] = invalid_cases

    # Valid case metrics
    tp_valid = valid_df[valid_df["_Presidio_Tag_Flag"] == 'true'].shape[0]
    fn_valid = valid_df[(valid_df["_Presidio_Tag_Flag"] == 'false_misanonymised') & (valid_df["_Presidio_entity_types"].astype(str) != "[]")].shape[0]
    value_tp_valid = valid_df[valid_df["_Presidio_Value_Flag"] == 'true'].shape[0]
    value_fn_valid = valid_df[(valid_df["_Presidio_Value_Flag"] == 'false_misanonymised') & (valid_df["_Presidio_entity_types"].astype(str) != "[]")].shape[0]
    leaked_valid = valid_df[valid_df["Presidio_Leaked_flag"] == True].shape[0]

    results[entity][2] = tp_valid
    results[entity][3] = fn_valid
    # (Valid) Tag mis-anonymisation distribution
    tag_fn_valid_df = valid_df[valid_df["_Presidio_Tag_Flag"] == 'false_misanonymised']
    results[entity][4] = get_mis_anonymisation_distribution(tag_fn_valid_df)

    results[entity][5] = value_tp_valid
    results[entity][6] = value_fn_valid
    # (Valid) Value mis-anonymisation distribution
    value_fn_valid_df = valid_df[valid_df["_Presidio_Value_Flag"] == 'false_misanonymised']
    results[entity][7] = get_mis_anonymisation_distribution(value_fn_valid_df)

    results[entity][8] = leaked_valid

    results[entity][9] = (tp_valid / valid_cases * 100) if valid_cases else 0
    results[entity][10] = (fn_valid / valid_cases * 100) if valid_cases else 0
    results[entity][11] = (value_tp_valid / valid_cases * 100) if valid_cases else 0
    results[entity][12] = (value_fn_valid / valid_cases * 100) if valid_cases else 0
    results[entity][13] = (leaked_valid / valid_cases * 100) if valid_cases else 0

    # Invalid case metrics
    tp_invalid = invalid_df[invalid_df["_Presidio_Tag_Flag"] == 'true'].shape[0]
    fn_invalid = invalid_df[(invalid_df["_Presidio_Tag_Flag"] == 'false_misanonymised') & (invalid_df["_Presidio_entity_types"].astype(str) != "[]")].shape[0]
    value_tp_invalid = invalid_df[invalid_df["_Presidio_Value_Flag"] == 'true'].shape[0]
    value_fn_invalid = invalid_df[(invalid_df["_Presidio_Value_Flag"] == 'false_misanonymised') & (invalid_df["_Presidio_entity_types"].astype(str) != "[]")].shape[0]
    leaked_invalid = invalid_df[invalid_df["Presidio_Leaked_flag"] == True].shape[0]

    results[entity][14] = tp_invalid
    results[entity][15] = fn_invalid
    # (Invalid) Tag mis-anonymisation distribution
    tag_fn_invalid_df = invalid_df[invalid_df["_Presidio_Tag_Flag"] == 'false_misanonymised']
    results[entity][16] = get_mis_anonymisation_distribution(tag_fn_invalid_df)

    results[entity][17] = value_tp_invalid
    results[entity][18] = value_fn_invalid
    # (Invalid) Value mis-anonymisation distribution
    value_fn_invalid_df = invalid_df[invalid_df["_Presidio_Value_Flag"] == 'false_misanonymised']
    results[entity][19] = get_mis_anonymisation_distribution(value_fn_invalid_df)

    results[entity][20] = leaked_invalid

    results[entity][21] = (tp_invalid / invalid_cases * 100) if invalid_cases else 0
    results[entity][22] = (fn_invalid / invalid_cases * 100) if invalid_cases else 0
    results[entity][23] = (value_tp_invalid / invalid_cases * 100) if invalid_cases else 0
    results[entity][24] = (value_fn_invalid / invalid_cases * 100) if invalid_cases else 0
    results[entity][25] = (leaked_invalid / invalid_cases * 100) if invalid_cases else 0

# Final column names
columns = [
    "Valid Occurrences", "Invalid Occurrences",
    "(tag) Valid - anonymised occurrences (TP)", "(tag) Valid - mis-anonymised occurrences (FN)",
    "(Valid)Tag Mis-anonymisation Distribution",
    "(value) Valid - anonymised", "(value) Valid - misanonymised",
    "(Valid)Value Mis-anonymisation Distribution",
    "Valid leaked occurrence",
    "(tag) Valid - anonymisation accuracy%", "(tag) Valid - misanonymisation accuracy%",
    "(value) Valid - anonymisation accuracy%", "(value) Valid - misanonymisation accuracy%", "Valid - Leakage %",
    "(tag) Invalid - anonymised occurrences (TP)", "(tag) Invalid - mis-anonymised occurrences (FN)",
    "(Invalid)Tag Mis-anonymisation Distribution",
    "(value) Invalid - anonymised", "(value) Invalid - misanonymised",
    "(Invalid)Value Mis-anonymisation Distribution",
    "Invalid leaked occurrence",
    "(tag) Invalid - anonymisation accuracy%", "(tag) Invalid - misanonymisation accuracy%",
    "(value) Invalid - anonymisation accuracy%", "(value) Invalid - misanonymisation accuracy%", "Invalid - Leakage %"
]

# Convert to DataFrame
output_df = pd.DataFrame.from_dict(results, orient='index', columns=columns)

# Save output
output_df.to_excel(output_file, index=True, index_label="Entity")

print("Processing complete. Output saved to:", output_file)
