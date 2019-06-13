import unicodedata


def parse(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as r:
        for line in r:
            for word in line.split("#")[1].split(","):
                word = unicodedata.normalize('NFKC', word.strip())
                yield word


def parse_lexemes(dict_file):
    with open(dict_file, 'r', encoding='utf-8') as r:
        for line in r:
            yield [unicodedata.normalize('NFKC', word.strip()) for word in line.split("#")[1].split(",")]