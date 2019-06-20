from typing import List

VOWELS = "aeiouAEIOUаоэиуыеёюяАОЭИУЫЕЁЮЯ"
CLOSED_SYLLABLE_CHARS = "рлймнРЛЙМН"


def get_first_vowel_position(string):
    for i, ch in enumerate(string):
        if ch in VOWELS:
            return i
    return -1


class Annotation:
    """
    Класс аннотации.
    Содержит начальную и конечную позицию в тексте, а также текст аннотации.
    """
    def __init__(self, begin: int, end: int, text: str) -> None:
        self.begin = begin
        self.end = end
        self.text = text


class Syllable(Annotation):
    """
    Разметка слога. Включает в себя аннотацию и номер слога, а также ударение.
    Если ударение падает не на этот слог, -1.
    """
    def __init__(self, begin: int, end: int, number: int, text: str, stress: int=-1) -> None:
        super(Syllable, self).__init__(begin, end, text)
        self.number = number
        self.stress = stress

    def vowel(self) -> int:
        """
        :return: позиция гласной буквы этого слога в слове (с 0).
        """
        return get_first_vowel_position(self.text) + self.begin

    def from_dict(self, d: dict) -> 'Syllable':
        self.__dict__.update(d)
        if "accent" in self.__dict__:
            self.stress = self.__dict__["accent"]
        return self


def get_syllables(word: str) -> List[Syllable]:
    """
    Разделение слова на слоги.
    :param word: слово для разбивки на слоги.
    :return syllables: массив слогов слова.
    """
    syllables = []
    begin = 0
    number = 0

    # В случае наличия дефиса разбиваем слова на подслова, находим слоги в них, объединяем.
    if "-" in word:
        word_parts = word.split("-")
        word_syllables = []
        last_part_end = 0
        for part in word_parts:
            part_syllables = get_syllables(part)
            if len(part_syllables) == 0:
                continue
            for i in range(len(part_syllables)):
                part_syllables[i].begin += last_part_end
                part_syllables[i].end += last_part_end
                part_syllables[i].number += len(word_syllables)
            word_syllables += part_syllables
            last_part_end = part_syllables[-1].end + 1
        return word_syllables

    # Для слов или подслов, в которых нет дефиса.
    for i, ch in enumerate(word):
        if ch not in VOWELS:
            continue
        if i + 1 < len(word) - 1 and word[i + 1] in CLOSED_SYLLABLE_CHARS:
            if i + 2 < len(word) - 1 and word[i + 2] in "ьЬ":
                # Если после сонорного согласного идёт мягкий знак, заканчиваем на нём. ("бань-ка")
                end = i + 3
            elif i + 2 < len(word) - 1 and word[i + 2] not in VOWELS and \
                    (word[i + 2] not in CLOSED_SYLLABLE_CHARS or word[i + 1] == "й"):
                # Если после сонорного согласного не идёт гласная или другой сонорный согласный,
                # слог закрывается на этом согласном. ("май-ка")
                end = i + 2
            else:
                # Несмотря на наличие закрывающего согласного, заканчиваем на гласной.
                # ("со-ло", "да-нный", "пол-ный")
                end = i + 1
        else:
            # Если после гласной идёт не закрывающая согласная, заканчиваем на гласной. ("ко-гда")
            end = i + 1
        syllables.append(Syllable(begin, end, number, word[begin:end]))
        number += 1
        begin = end
    if get_first_vowel_position(word) != -1:
        # Добиваем последний слог до конца слова.
        syllables[-1] = Syllable(syllables[-1].begin, len(word), syllables[-1].number,
                                 word[syllables[-1].begin:len(word)])
    return syllables
