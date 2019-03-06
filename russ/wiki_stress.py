import unicodedata
import re

from russ.settings import RU_WIKTIONARY_DICT


def parse(dump_path, phrases_mode):
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
                word = word.replace(chr(1117), "и" + chr(768)).replace(chr(1104), "е" + chr(768))
                has_good_only = True
                for ch in word:
                    if not ('а' <= ch <= 'я' or 'А' <= ch <= 'Я' or ord(ch) == 769 or ord(ch) == 768 or ch in "()~-ёЁ "):
                        has_good_only = False
                if not has_good_only:
                    continue
                elif "(" in word and ")" not in word or ")" in word and "(" not in word:
                    continue
                elif "~" in word:
                    for w in word.split(" ~ "):
                        if w and not phrases_mode and "~" not in w:
                            yield w
                elif " " in word:
                    if phrases_mode:
                        yield word
                    else:
                        continue
                elif word and not phrases_mode:
                    yield word


def save(dump_path, dict_path=RU_WIKTIONARY_DICT, phrases_mode=False):
    with open(dict_path, "w", encoding="utf-8") as w:
        for word in parse(dump_path, phrases_mode):
            word = word.strip().replace(chr(769), chr(39)).replace(chr(768), chr(96))
            matches = re.findall(r'\(\w+\)', word)
            if len(matches) == 1:
                w.write(word.replace(matches[0], "") + "\n")
            word = word.replace("(", "").replace(")", "")
            w.write(word + "\n")
