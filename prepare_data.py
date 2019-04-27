import argparse
import random
import unicodedata

from russ.parsers.inflected_stress import parse as parse_inflected_stress
from russ.parsers.wiki_stress import parse as parse_wiki_stress


def prepare(wiktionary_dump_path: str, inflected_dict_path: str,
            custom_dict_path: str, result_path: str,
            sort: bool=False, shuffle: bool=True, lower: bool=False):
    assert sort != shuffle
    words = set()
    for word in parse_inflected_stress(inflected_dict_path):
        words.add(word.strip())
    for word in parse_wiki_stress(wiktionary_dump_path):
        words.add(word.strip())
    with open(custom_dict_path, "r", encoding="utf-8") as f:
        for word in f:
            word = unicodedata.normalize('NFKC', word.strip())
            words.add(word.strip())
    words = list(words)
    if lower:
        words = list(map(str.lower, words))
    if sort:
        words.sort()
    if shuffle:
        random.shuffle(words)
    with open(result_path, "w", encoding="utf-8") as f:
        for word in words:
            if not word or word[0] in (chr(39), chr(96), '-'):
                continue
            f.write(word.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiktionary-dump-path', required=True)
    parser.add_argument('--inflected-dict-path', required=True)
    parser.add_argument('--custom-dict-path', default=None)
    parser.add_argument('--result-path', required=True)
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lower', action='store_true')
    args = parser.parse_args()
    prepare(**vars(args))
