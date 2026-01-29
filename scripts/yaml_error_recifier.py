from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

yaml = YAML()
yaml.preserve_quotes = True

file_path = "yaml/recognizer_hr_config.yaml"  # Update path if needed

def check_for_missing_regex(data, path="root"):
    if isinstance(data, list):
        for i, item in enumerate(data):
            check_for_missing_regex(item, f"{path}[{i}]")
    elif isinstance(data, dict) or isinstance(data, CommentedMap):
        if 'name' in data and 'regex' not in data:
            line = getattr(data, 'lc', None)
            line_no = line.line + 1 if line else "?"
            print(f"❌ Missing 'regex' at {path}, line {line_no}: {data.get('name', 'Unknown name')}")
        for key, value in data.items():
            check_for_missing_regex(value, f"{path}.{key}")

with open(file_path, "r") as f:
    data = yaml.load(f)

check_for_missing_regex(data)
