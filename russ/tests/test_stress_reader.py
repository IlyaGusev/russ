import unittest
import os
from tempfile import NamedTemporaryFile

from russ.stress.reader import StressReader


class TestStressReader(unittest.TestCase):
    def test_reader(self):
        reader = StressReader()
        file = NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        file.write("прищу'чу\nперемеша'ют\nдуше`внобольны'х\nэлектро`н-во'льтом")
        file.close()
        dataset = reader.read(file.name)
        labels = [
            ['0', '0', '0', '0', '0', '1', '0', '0', '0'],
            ['0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0'],
            ['0', '0', '0', '0', '2', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0'],
            ['0', '0', '0', '0', '0', '0', '0', '2', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0'],
        ]
        for l, sample in zip(labels, dataset):
            self.assertListEqual(sample["tags"].labels, l)
        os.unlink(file.name)
