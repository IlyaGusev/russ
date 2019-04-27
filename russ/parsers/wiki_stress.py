import unicodedata
import re


def clean_word(word):
    word = word.strip().replace(chr(769), chr(39)).replace(chr(768), chr(96))
    matches = re.findall(r'\(\w+\)', word)
    if len(matches) == 1:
        yield word.replace(matches[0], "")
    yield word.replace("(", "").replace(")", "")


def parse(dump_path, phrases_mode=False):
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
                        if not w or phrases_mode or "~" in w:
                            continue
                        for w2 in clean_word(w):
                            yield w2
                elif " " in word:
                    if not phrases_mode:
                        continue
                    for w2 in clean_word(word):
                        yield w2
                elif word and not phrases_mode:
                    for w2 in clean_word(word):
                        yield w2
