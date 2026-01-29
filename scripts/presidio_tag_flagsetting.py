import pandas as pd
import ast

# Load the Excel file
file_path = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Montenegrin_specific_outputs.xlsx"
df = pd.read_excel(file_path)

# Ensure proper conversion of '_Presidio_entity_types' if it is stored as string
def parse_entity_list(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except:
        return []

df['_Presidio_entity_types'] = df['_Presidio_entity_types'].apply(parse_entity_list)

# Entity mapping dictionary
entity_mapping = {
    "CUSTOMER_ACCOUNT": ["CUSTOMER_NR"],
    "CUSTOMER": ["CUSTOMER_NR", "AUSTRIAN_CUSTOMER_NUMBER"],
    "PASSPORT": [
        "INDIAN_PASSPORT", "US_PASSPORT", "ITALY_PASSPORT", "CANADA_PASSPORT",
        "FRANCE_PASSPORT", "GERMAN_PASSPORT", "SWEDEN_PASSPORT", "UK_PASSPORT",
        "AUSTRIA_PASSPORT", "RUSSIAN_PASSPORT", "CROATIAN_PASSPORT", "HUNGARY_PASSPORT"
    ],
    "TAX_IDENTIFICATION_NUMBER": ["TAX_IDENTIFICATION_NR"],
    "BILLING_NR": ["BILLING_NUMBER"],
    "CREDIT_CARD": ["CREDIT_CARD_NR"],
    "IBAN": ["IBAN_CODE"],
    "IP_ADDRESS": ["NUMBER"]

}

# Function to compute _Presidio_Tag_Flag with mapping consideration
def compute__Presidio_Tag_Flag(row):
    entity = row['Entity']
    detected_entities = row['_Presidio_entity_types']
    # Direct match
    if entity in detected_entities:
        return 'true'
    # Mapped match
    if entity in entity_mapping:
        for alt in entity_mapping[entity]:
            if alt in detected_entities:
                return 'true'
    # Else decide based on leakage
    return 'false_leaked' if row['Presidio_Leaked_flag'] else 'false_misanonymised'

# Apply tagging
df['_Presidio_Tag_Flag'] = df.apply(compute__Presidio_Tag_Flag, axis=1)

# Get list of unique entities
unique_entities = df['Entity'].unique()

# Print stats per entity
for entity in unique_entities:
    entity_df = df[df['Entity'] == entity]
    total = len(entity_df)
    correct = (entity_df['_Presidio_Tag_Flag'] == 'true').sum()
    mis = (entity_df['_Presidio_Tag_Flag'] == 'false_misanonymised').sum()
    leak = (entity_df['_Presidio_Tag_Flag'] == 'false_leaked').sum()

    accuracy = (correct / total) * 100 if total else 0
    leakage_percentage = (leak / total) * 100 if total else 0

    print(f"Entity: {entity}")
    print(f"  Total Count: {total}")
    print(f"  Correct Anonymization Count: {correct}")
    print(f"  Mis-Anonymization Count: {mis}")
    print(f"  Leakage Count: {leak}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Leakage Percentage: {leakage_percentage:.2f}%\n")

# Overall stats
total_records = len(df)
total_correct = (df['_Presidio_Tag_Flag'] == 'true').sum()
total_mis = (df['_Presidio_Tag_Flag'] == 'false_misanonymised').sum()
total_leak = (df['_Presidio_Tag_Flag'] == 'false_leaked').sum()

overall_correct_acc = (total_correct / total_records) * 100 if total_records else 0
overall_mis_acc = (total_mis / total_records) * 100 if total_records else 0
overall_leakage = (total_leak / total_records) * 100 if total_records else 0

print("Overall Statistics:")
print(f"  Total Records: {total_records}")
print(f"  Correct Anonymization Count: {total_correct}")
print(f"  Mis-Anonymization Count: {total_mis}")
print(f"  Leakage Count: {total_leak}")
print(f"  Correct Anonymization Accuracy: {overall_correct_acc:.2f}%")
print(f"  Mis-Anonymization Accuracy: {overall_mis_acc:.2f}%")
print(f"  Overall Leakage Percentage: {overall_leakage:.2f}%")

# Save the updated DataFrame back to the same Excel file
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, index=False)

print("\nUpdated file saved with '_Presidio_Tag_Flag' column.")
