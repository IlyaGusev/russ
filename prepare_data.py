import argparse
import random
import unicodedata

from russ.parsers.inflected_stress import parse as parse_inflected_stress
from russ.parsers.inflected_stress import parse_lexemes
from russ.parsers.wiki_stress import parse as parse_wiki_stress


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

    words = list(words)
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
                if not word or word[0] in (chr(39), chr(96), '-'):
                    continue
                choose_file(count).write(word.strip() + "\n")
                count += 1
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
