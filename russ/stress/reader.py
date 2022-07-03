def text_to_instance(text: str):
    clean_word = ""
    schema = []
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
    tokens = list(clean_word)

    result = dict()
    result["tokens"] = tokens
    if schema:
        result["tags"] = schema
    return result
