import random
from russ.settings import RU_INFLECTED_DICT, RU_WIKTIONARY_DICT, RU_CUSTOM_DICT, RU_ALL_DICT


def join(input_paths=(RU_INFLECTED_DICT, RU_WIKTIONARY_DICT, RU_CUSTOM_DICT), output_path=RU_ALL_DICT, is_lower=False):
    words = []
    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                words.append(line.strip())
    words = list(set(words))
    random.shuffle(words)
    with open(output_path, "w", encoding="utf-8") as f:
        for word in words:
            if is_lower:
                word = word.lower()
            if not word or word[0] == chr(39) or word[0] == chr(96):
                continue
            f.write(word.strip() + "\n")

join()