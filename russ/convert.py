from typing import Dict
from russ.syllables import VOWELS

def convert_to_record(text: str) -> Dict:
    vowels = set(VOWELS)
    text = text.strip().replace("ё'", "ё").replace("ё", "ё'")
    clean_word, schema = str(), list()
    for i, ch in enumerate(text):
        prev_ch = text[i - 1] if i >= 1 else None
        if ch == "'" and i > 0 and prev_ch in vowels:
            schema[-1] = 1
        elif ch == "`" and i > 0 and prev_ch in vowels:
            schema[-1] = 2
        elif ch == "ё":
            schema.append(1)
            clean_word += ch
        else:
            schema.append(0)
            clean_word += ch
    assert len(schema) == len(clean_word)
    return {
        "tokens": list(clean_word),
        "text": clean_word,
        "tags": schema
    }
