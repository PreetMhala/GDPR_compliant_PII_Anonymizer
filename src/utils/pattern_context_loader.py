import yaml
from presidio_analyzer import Pattern 
from .constants import *
def pattern_context_read(key1,key2,key3):
    with open(ACTIVE_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) 
        custom_data=data[key1]
        PATTERNS=[Pattern(name=entry[NAME], regex=entry[REGEX], score=entry[CONFIDENCE]) for entry in custom_data.get(key2, [])]
        CONTEXT=data.get(key3,[])
    return PATTERNS,CONTEXT