import unicodedata

from russ.settings import RU_INFLECTED_DICT


def parse(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as r:
        for line in r:
            for word in line.split("#")[1].split(","):
                word = unicodedata.normalize('NFKC', word.strip())
                yield word


def save(original_dict_path, dict_path=RU_INFLECTED_DICT):
    with open(dict_path, "w", encoding="utf-8") as w:
        for word in parse(original_dict_path):
            word = word.strip()
            w.write(word + "\n")
