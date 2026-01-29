import pandas as pd

# File paths
file1_path = "/Users/pritmhala/Downloads/PII_Detector_base_ver2 2/Greek_new_data_entity_specific_raw.xlsx"
file2_path = "/Users/pritmhala/Downloads/PII_Detector_base_ver2 2/test/croation_entity_specific_data_raw.xlsx"

# Load the Excel files
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# Fill the Validity column in df1 by matching the Query with Original_Query in df2
def get_validity(query):
    match = df2[df2['Original_Query'] == query]
    if not match.empty:
        return match['Validity'].values[0]
    return ''  # Leave blank if no match is found

df1['Validity'] = df1['Query'].apply(get_validity)

# Save the updated file
output_path = file1_path.replace(".xlsx", "_with_validity.xlsx")
df1.to_excel(output_path, index=False)

print(f"Updated file saved to: {output_path}")
