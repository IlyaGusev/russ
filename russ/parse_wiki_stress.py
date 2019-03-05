import unicodedata

from russ.settings import RU_WIKTIONARY_DICT


def parse(dump_path):
    with open(dump_path, "r", encoding="utf-8") as r:
        count = 0
        for line in r:
            if "transcriptions-ru" not in line and "transcription-ru" not in line:
                continue
            count += 1
            if "}}" not in line or "{{" not in line:
                continue
            line = line.strip()[2:].split("}}")[0]
            if "}}" in line or "{{" in line:
                continue
            parts = line.split("|")[:3]
            words = parts[1:]
            for word in words:
                word = unicodedata.normalize('NFKC', word.strip())
                if "~" in word:
                    for w in word.split(" ~ "):
                        if w:
                            yield w
                elif " " in word or "=" in word or len(word) <= 2 or "ogg" in word or "̏" in word:
                    continue
                elif word:
                    yield word


def save_dict(dump_path, dict_path=RU_WIKTIONARY_DICT):
    with open(dict_path, "w", encoding="utf-8") as w:
        for word in parse(dump_path):
            word = word.strip()
            w.write(word + "\n")


save_dict("/media/yallen/My Passport/Datasets/ruwiktionary-20190301-pages-articles.xml")
