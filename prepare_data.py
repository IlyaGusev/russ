import argparse
import random
import unicodedata
from collections import defaultdict

from russ.parsers.inflected_stress import parse as parse_inflected_stress
from russ.parsers.inflected_stress import parse_lexemes
from russ.parsers.wiki_stress import parse as parse_wiki_stress
from russ.stress.reader import StressReader


def merge_forms(words):
    clean_to_marked = defaultdict(set)
    for word in words:
        word = word.replace("ё'", "ё").replace("ё", "ё'")
        clean_word = word.replace(chr(39), "").replace(chr(96), "")
        if not word or word == clean_word or word[0] in (chr(39), chr(96), '-'):
            continue
        clean_to_marked[clean_word].add(word)
    words = []
    reader = StressReader()
    for clean_word, variants in clean_to_marked.items():
        if len(variants) == 1:
            words.append(variants.pop())
            continue
        primary = set()
        secondary = set()
        for word in variants:
            schema = list(map(int, reader.text_to_instance(word)["tags"].labels))
            primary |= {i-1 for i, stress in enumerate(schema) if stress == 1}
            secondary |= {i-1 for i, stress in enumerate(schema) if stress == 2}
        secondary = secondary - primary
        if not primary and not secondary:
            continue
        word = ""
        for i, ch in enumerate(clean_word):
            word += ch
            if i in primary:
                word += chr(39)
            elif i in secondary:
                word += chr(96)
        words.append(word)
    return words


def prepare(wiktionary_dump_path: str,
            inflected_dict_path: str,
            custom_dict_path: str,
            train_path: str,
            val_path: str,
            test_path: str,
            val_part: float,
            test_part: float,
            split_lexemes: bool=False,
            sort: bool=False,
            shuffle: bool=True,
            lower: bool=False):
    assert sort != shuffle
    # assert not split_lexemes or inflected_dict_path and not wiktionary_dump_path and not custom_dict_path

    lexemes = []
    words = set()

    if inflected_dict_path and split_lexemes:
        for lexeme in parse_lexemes(inflected_dict_path):
            lexemes.append(lexeme)
    else:
        if inflected_dict_path:
            for word in parse_inflected_stress(inflected_dict_path):
                words.add(word.strip())
        if wiktionary_dump_path:
            for word in parse_wiki_stress(wiktionary_dump_path):
                words.add(word.strip())
        if custom_dict_path:
            with open(custom_dict_path, "r", encoding="utf-8") as f:
                for word in f:
                    word = unicodedata.normalize('NFKC', word.strip())
                    words.add(word.strip())

    words = merge_forms(words)
    if lower:
        words = list(map(str.lower, words))
        lexemes = map(lambda x: map(str.lower, x), lexemes)
    if sort:
        words.sort()
        lexemes.sort(key=lambda x: x[0])
    if shuffle:
        random.shuffle(words)
        random.shuffle(lexemes)

    all_count = len(words) if not split_lexemes else len(lexemes)
    print(all_count)
    count = 0
    val_border = 1. - val_part - test_part
    test_border = 1. - test_part
    with open(train_path, "w", encoding="utf-8") as train,\
            open(val_path, "w", encoding="utf-8") as val,\
            open(test_path, "w", encoding="utf-8") as test:

        def choose_file(count):
            current_file = train
            if val_border < float(count) / all_count < test_border:
                current_file = val
            elif float(count) / all_count >= test_border:
                current_file = test
            return current_file

        if not split_lexemes:
            for word in words:
                count += 1
                choose_file(count).write(word.strip() + "\n")
        else:
            for lexeme in lexemes:
                for word in lexeme:
                    if not word or word[0] in (chr(39), chr(96), '-'):
                        continue
                    choose_file(count).write(word.strip() + "\n")
                count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiktionary-dump-path', default=None)
    parser.add_argument('--inflected-dict-path', default=None)
    parser.add_argument('--custom-dict-path', default=None)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--val-path', required=True)
    parser.add_argument('--val-part', type=float, default=0.05)
    parser.add_argument('--test-part', type=float, default=0.05)
    parser.add_argument('--split-lexemes', action='store_true')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lower', action='store_true')
    args = parser.parse_args()
    prepare(**vars(args))
