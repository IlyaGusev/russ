def text_to_instance(text: str):
    clean_word, schema = str(), list()
    for ch in text:
        if ch == "'":
            schema[-1] = 1
        elif ch == "`":
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
        "tags": schema
    }
