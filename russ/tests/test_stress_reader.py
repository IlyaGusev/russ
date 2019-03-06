import unittest
import os
from tempfile import NamedTemporaryFile

from russ.stress_model.stress_reader import StressReader


class TestStressReader(unittest.TestCase):
    def test_reader(self):
        reader = StressReader()
        file = NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
        file.write("прищу'чу\nперемеша'ют\nдуше`внобольны'х\nэлектро`н-во'льтом")
        file.close()
        dataset = reader.read(file.name)
        os.unlink(file.name)
        for sample in dataset:
            print(sample)