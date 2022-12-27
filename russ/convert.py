from typing import Dict

from russ.syllables import VOWELS

PRIMARY_STRESS_CHAR = "'"
SECONDARY_STRESS_CHAR = "`"


def convert_to_record(text: str) -> Dict:
    vowels = set(VOWELS)
    text = text.strip()
    text = text.replace("ё" + PRIMARY_STRESS_CHAR, "ё")
    text = text.replace("ё" + SECONDARY_STRESS_CHAR, "ё")
    text = text.replace("ё", "ё" + PRIMARY_STRESS_CHAR)
    clean_word, schema = str(), list()
    for i, ch in enumerate(text):
        prev_ch = text[i - 1] if i >= 1 else None
        if ch == PRIMARY_STRESS_CHAR and i > 0 and prev_ch in vowels:
            schema[-1] = 1
        elif ch == SECONDARY_STRESS_CHAR and i > 0 and prev_ch in vowels:
            schema[-1] = 2
        else:
            schema.append(0)
            clean_word += ch
    assert len(schema) == len(clean_word)
    return {
        "tokens": list(clean_word),
        "text": clean_word,
        "tags": schema
    }
