import unicodedata

from russ.syllables import VOWELS
from russ.convert import PRIMARY_STRESS_CHAR, SECONDARY_STRESS_CHAR


def get_first_vowel_position(string):
    for i, ch in enumerate(string):
        if ch in VOWELS:
            return i
    return -1


def parse(dict_file):
    vowels = set(VOWELS)
    with open(dict_file, 'r', encoding='utf-8') as r:
        for line in r:
            word, stress_str = line.strip().split()
            stress_vowel_index = int(stress_str.replace("$", ""))
            word = unicodedata.normalize('NFKC', word.strip())
            vowel_index = 0
            converted_word = ""
            for ch in word:
                converted_word += ch
                if ch not in vowels:
                    continue
                vowel_index += 1
                if vowel_index != stress_vowel_index:
                    continue
                converted_word += PRIMARY_STRESS_CHAR
            yield converted_word
