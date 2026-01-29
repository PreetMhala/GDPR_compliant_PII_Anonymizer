import pandas as pd
from collections import defaultdict

# Define the numerical placeholders
numerical_placeholders = {
    "PHONE_NUMBER": "0000000000",
    "CUSTOMER_NR": "0000000000",
    "CUSTOMER": "0000000000",
    "CUSTOMER_ACCOUNT": "0000000000",
    "AUSTRIAN_CUSTOMER_NUMBER": "0000000000",
    "CREDIT_CARD_NR": "<BIG_NUMBER>",
    "CREDIT_CARD": "<BIG_NUMBER>",
    "ZIP_CODE": "00000",
    "SSN": "000-00-0000",
    "IN_PAN": "0000000000",
    "AU_TFN": "000000000",
    "BILLING_NR": "<BIG_NUMBER>",
    "ROUTING_NUMBER": "000000000",
    "NRP": "000000",
    "BIG_NUMBERS": "<BIG_NUMBER>",
    "DRIVER_LICENSE": "0000000000",
    "SWIFT_CODE": "00000000",
    "AU_ACN": "000000000",
    "LICENSE_PLATE": "0000000000",
    "LICENSE_PLATE_NUMBER": "00000000",
    "ID": "0000000000",
    "IBAN": "<BIG_NUMBER>",
    "IN_AADHAAR": "<BIG_NUMBER>",
    "PASSPORT": "0000000000",
    "INDIAN_PASSPORT": "0000000000",
    "US_PASSPORT": "0000000000",
    "ITALY_PASSPORT": "0000000000",
    "CANADA_PASSPORT": "0000000000",
    "FRANCE_PASSPORT": "0000000000",
    "GERMAN_PASSPORT": "0000000000",
    "SWEDEN_PASSPORT": "0000000000",
    "UK_PASSPORT": "0000000000",
    "AUSTRIA_PASSPORT": "0000000000",
    "RUSSIAN_PASSPORT": "0000000000",
    "POLAND_PASSPORT": "0000000000",
    "GREEK_PASSPORT": "0000000000",
    "GERMAN_ID": "<BIG_NUMBER>",
    "POLISH_ID": "<BIG_NUMBER>",
    "ROMANIAN_ID": "<BIG_NUMBER>",
    "MEDICAL_LICENSE": "00000000",
    "UK_NHS": "0000000000",
    "GERMAN_PRE_ID": "0000000000",
    "GERMAN_SSN": "<BIG_NUMBER>",
    "TAX_IDENTIFICATION_NR": "<BIG_NUMBER>",
    "TAX_IDENTIFICATION_NUMBER": "0000000000",
    "STREET": "<SOME_NAME>",
    "ORGANIZATION": "<SOME_NAME>",
    "PERSON": "<SOME_NAME>",
    "EMAIL_ADDRESS": "<EMAIL_ADDRESS>",
    "INTERNET_ACCESS_NUMBER": "<EMAIL_ADDRESS>",
    "URL": "<URL>",
    "SOCIAL_MEDIA_HANDLE": "<@SOME_NAME>",
    "GPS_COORDINATES": "0000000000",
    "DATE": "<DATE_TIME>",
    "CARD_PROFILE_NR": "<BIG_NUMBER>",
    "CRYPTO": "<BIG_NUMBER>",
    "IMEI": "<BIG_NUMBER>",
    "OIB": "<BIG_NUMBER>",
    "CROATIAN_VISA": "00000000",
    "GERMAN_VISA": "000000000",
    "POLISH_TAX_ID_NIP": "0000000000",
    "POLISH_VISA": "000000000",
    "PESEL": "<BIG_NUMBER>",
    "US_DRIVER_LICENSE": "00000000",
    "AU_MEDICARE": "000000000",
    "AU_ABN": "<BIG_NUMBER>",
    "GREECE_VISA": "000000000",
    "GREECE_HEALTH_ID_NUMBER": "<BIG_NUMBER>",
    "HUNGARIAN_VISA": "000000000",
    "HUNGARIAN_LICENSE_PLATE_NUMBER": "00000000",
    "HUNGARIAN_DOMESTIC_BANK_ACCOUNT": "<BIG_NUMBER>",
    "HUNGARIAN_TAJ_NUMBER": "<BIG_NUMBER>"
}



# Load Excel file
file_path = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Montenegrin_specific_outputs.xlsx"
df = pd.read_excel(file_path)

# Initialize counters
correct_counts = defaultdict(int)
mis_counts = defaultdict(int)
leakage_counts = defaultdict(int)
total_counts = defaultdict(int)

def evaluate_anonymization_status(row):
    entity = row["Entity"]
    anonymized = row["_Presidio_mask_query"]
    leaked = row["Presidio_Leaked_flag"]
    expected_placeholder = numerical_placeholders.get(entity)

    # Track total counts for each entity
    total_counts[entity] += 1

    # Step 3: If the row is a leak, mark as 'false_leaked'
    if leaked:  # Leaked_flag is True
        leakage_counts[entity] += 1
        return 'false_leaked'

    # Step 1: Check for leaks based on the Leaked_flag
    if pd.isna(leaked) or leaked == [] or not leaked:  # Leaked_flag is False or missing
        # Step 2: Check for correct anonymization (if Leaked_flag is False or missing)
        if isinstance(expected_placeholder, str) and expected_placeholder in anonymized:
            correct_counts[entity] += 1
            return 'true'
        else:
            mis_counts[entity] += 1
            return 'false_misanonymised'

    return None




# Apply function to DataFrame
df["_Presidio_Value_Flag"] = df.apply(evaluate_anonymization_status, axis=1)

# Save updated file
df.to_excel(file_path, index=False)

# Print results
print("Entity Anonymization Analysis:\n")
total_correct = sum(correct_counts.values())
total_mis = sum(mis_counts.values())
total_leaks = sum(leakage_counts.values())
total_records = len(df)

for entity in total_counts.keys():
    total = total_counts[entity]
    correct = correct_counts[entity]
    mis = mis_counts[entity]
    leaks = leakage_counts[entity]
    accuracy = (correct / total) * 100 if total > 0 else 0
    leakage_percentage = (leaks / total) * 100 if total > 0 else 0
    print(f"Entity: {entity}")
    print(f"  Total Count: {total}")
    print(f"  Correct Anonymization Count: {correct}")
    print(f"  Mis-Anonymization Count: {mis}")
    print(f"  Leakage Count: {leaks}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Leakage Percentage: {leakage_percentage:.2f}%\n")

# Overall summary
overall_accuracy = (total_correct / total_records) * 100 if total_records > 0 else 0
mis_accuracy = (total_mis / total_records) * 100 if total_records > 0 else 0
overall_leakage_percentage = (total_leaks / total_records) * 100 if total_records > 0 else 0

print("Overall Statistics:")
print(f"  Total Records: {total_records}")
print(f"  Correct Anonymization Count: {total_correct}")
print(f"  Mis-Anonymization Count: {total_mis}")
print(f"  Leakage Count: {total_leaks}")
print(f"  Correct Anonymization Accuracy: {overall_accuracy:.2f}%")
print(f"  Mis-Anonymization Accuracy: {mis_accuracy:.2f}%")
print(f"  Overall Leakage Percentage: {overall_leakage_percentage:.2f}%")