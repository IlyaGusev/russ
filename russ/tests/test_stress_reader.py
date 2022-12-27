import unittest
import os

from russ.convert import convert_to_record


class TestStressReader(unittest.TestCase):
    def test_reader(self):
        words = [
            "прищу'чу",
            "перемеша'ют",
            "душе`внобольны'х",
            "электро`н-во'льтом"
        ]
        all_labels = [
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        ]

        for word, labels in zip(words, all_labels):
            self.assertListEqual(convert_to_record(word)["tags"], labels)
