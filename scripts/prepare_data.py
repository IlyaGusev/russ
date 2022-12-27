import argparse
import random
import unicodedata
from tqdm import tqdm
from collections import defaultdict

from pymorphy2 import MorphAnalyzer

from russ.parsers.inflected_stress import parse as parse_inflected_stress
from russ.parsers.wiki_stress import parse as parse_wiki_stress
from russ.parsers.espeak_stress import parse as parse_espeak_stress
from russ.syllables import VOWELS, get_syllables
from russ.convert import convert_to_record


PRIMARY_STRESS = chr(39)
SECONDARY_STRESS = chr(96)
SPEC_SYMBOLS = (PRIMARY_STRESS, SECONDARY_STRESS, "-")
MORPH = MorphAnalyzer()


def merge_forms(words):
    clean_to_marked = defaultdict(set)
    for word in tqdm(words, desc="Merge 1"):
        if not word or word[0] in SPEC_SYMBOLS:
            continue
        clean_word = convert_to_record(word)["text"]
        if word == clean_word:
            continue
        clean_to_marked[clean_word].add(word)

    words = []
    vowels = set(VOWELS)
    for clean_word, variants in tqdm(clean_to_marked.items(), desc="Merge 2"):
        primary, secondary = set(), set()
        for word in variants:
            schema = list(map(int, convert_to_record(word)["tags"]))
            primary |= {i for i, stress in enumerate(schema) if stress == 1}
            secondary |= {i for i, stress in enumerate(schema) if stress == 2}
        secondary = secondary - primary
        if not primary:
            continue

        word = ""
        count_primary = 0
        for i, ch in enumerate(clean_word):
            word += ch
            if i in primary:
                assert word[-1] in vowels
                word += PRIMARY_STRESS
                count_primary += 1
            elif i in secondary:
                assert word[-1] in vowels
                word += SECONDARY_STRESS

        if count_primary != 0:
            words.append(word)
    return words


def merge_lexemes(words):
    lexemes = defaultdict(list)
    for word in tqdm(words, desc="Merging lexemes"):
        clean_word = convert_to_record(word)["text"]
        normal_form = MORPH.parse(clean_word)[0].normal_form
        lexemes[normal_form].append(word)
    return lexemes


def prepare(
    wiktionary_dump_path: str,
    inflected_dict_path: str,
    custom_dict_path: str,
    espeak_dict_path: str,
    inflected_sample_rate: float,
    all_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    val_part: float,
    test_part: float,
    split_mode: str = "lexemes",
    lower: bool = False,
    seed: int = 42
):
    assert split_mode in ("lexemes", "sort", "shuffle")
    random.seed(seed)

    words = set()
    if espeak_dict_path:
        for word in tqdm(parse_espeak_stress(espeak_dict_path), desc="Espeak"):
            words.add(word.strip())
    if inflected_dict_path:
        for word in tqdm(parse_inflected_stress(inflected_dict_path), desc="Inflected"):
            if random.random() < inflected_sample_rate:
                words.add(word.strip())
    if wiktionary_dump_path:
        for word in tqdm(parse_wiki_stress(wiktionary_dump_path), desc="Wiktionary"):
            words.add(word.strip())
    if custom_dict_path:
        with open(custom_dict_path, "r", encoding="utf-8") as f:
            for word in tqdm(f, desc="Custom"):
                word = unicodedata.normalize("NFKC", word.strip())
                words.add(word.strip())

    words = [w for w in tqdm(words, desc="Filtering") if len(get_syllables(w)) > 1]

    words = merge_forms(words)
    if lower:
        words = list(map(str.lower, words))
    if split_mode == "sort":
        words.sort()
    elif split_mode == "shuffle":
        random.shuffle(words)
    elif split_mode == "lexemes":
        lexemes = merge_lexemes(words)
        words = []
        for nf, lexeme in tqdm(lexemes.items(), desc="Extending lexemes"):
            words.extend(lexeme)

    all_count = len(words)
    print(all_count)
    count = 0
    val_border = 1. - val_part - test_part
    test_border = 1. - test_part
    with open(train_path, "w") as train, open(val_path, "w") as val, open(test_path, "w") as test, open(all_path, "w") as allf:
        def choose_file(count):
            current_file = train
            if val_border < float(count) / all_count < test_border:
                current_file = val
            elif float(count) / all_count >= test_border:
                current_file = test
            return current_file

        for word in words:
            count += 1
            choose_file(count).write(word.strip() + "\n")
            allf.write(word.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiktionary-dump-path", default=None)
    parser.add_argument("--espeak-dict-path", default=None)
    parser.add_argument("--inflected-dict-path", default=None)
    parser.add_argument("--inflected-sample-rate", default=0.3, type=float)
    parser.add_argument("--custom-dict-path", default=None)
    parser.add_argument("--all-path", default="data/all.txt")
    parser.add_argument("--train-path", default="data/train.txt")
    parser.add_argument("--test-path", default="data/test.txt")
    parser.add_argument("--val-path", default="data/val.txt")
    parser.add_argument("--val-part", type=float, default=0.05)
    parser.add_argument("--test-part", type=float, default=0.05)
    parser.add_argument("--split-mode", choices=("lexemes", "sort", "shuffle"), default="lexemes")
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    prepare(**vars(args))
