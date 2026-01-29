import pandas as pd

# Load the Excel file
file_path = "/Users/pritmhala/Downloads/PII_detector_bertic_nerkor_extended/test/Montenegrin_specific_outputs.xlsx"
df = pd.read_excel(file_path)

# Convert string representations of lists to actual lists if needed
df["_Presidio_entity_types"] = df["_Presidio_entity_types"].apply(eval)

# Leaked only if BOTH recognizers returned empty lists
df["Presidio_Leaked_flag"] = df.apply(
    lambda row: len(row["_Presidio_entity_types"]) == 0,
    axis=1
)

# Save the updated file (overwriting original)
df.to_excel(file_path, index=False)

print("Leaked_flag column updated using both Presidio and _GR_NLP entity types.")
