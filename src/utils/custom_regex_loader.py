import yaml
from .constants import *
def load_patterns_from_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return [
        (item[NAME], item[REGEX], item.get(CONFIDENCE, 1.0), item.get(KEYWORDS, []))
        for item in data[PATTERN_CUSTOM][PATTERNS]
    ]
