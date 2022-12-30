from typing import Dict

from russ.syllables import VOWELS

PRIMARY_STRESS_CHAR = "'"
SECONDARY_STRESS_CHAR = "`"


def convert_to_record(
    text: str,
    skip_secondary: bool = False,
    convert_secondary: bool = False
) -> Dict:
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
    if skip_secondary:
        schema = [(0 if s == 2 else s) for s in schema]
    if convert_secondary:
        schema = [(1 if s == 2 else s) for s in schema]
    return {
        "tokens": list(clean_word),
        "text": clean_word,
        "tags": schema
    }
