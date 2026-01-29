from gr_nlp_toolkit import Pipeline

# Instantiate the NER pipeline
nlp = Pipeline("ner")

# Function to replace entities with tags
def replace_entities_with_tags(text):
    doc = nlp(text)
    tokens = doc.tokens
    reconstructed = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        ner_tag = token.ner
        word = token.text
        if ner_tag.startswith("B-"):
            entity_type = ner_tag[2:]
            i += 1
            while i < len(tokens) and tokens[i].ner.startswith("I-"):
                i += 1
            if i < len(tokens) and tokens[i].ner.startswith("E-"):
                i += 1
            reconstructed.append(f"<{entity_type}>")
        elif ner_tag.startswith("S-"):
            entity_type = ner_tag[2:]
            reconstructed.append(f"<{entity_type}>")
            i += 1
        else:
            reconstructed.append(word)
            i += 1
    return ' '.join(reconstructed)

# Accept input from the user
input_text = input(" Enter a Greek sentence for NER tagging: ")

# Process and print result
result = replace_entities_with_tags(input_text)
print(f"\n Tagged output:\n{result}")
