# -*- coding: utf-8 -*-
# Author: Ilya Gusev
# Description: Stress model predictions test

import unittest

from russ.stress.predictor import StressPredictor, PredictSchema


class TestStressPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = StressPredictor("IlyaGusev/ru-word-stress-transformer")

    def test_stress(self):
        checks = {
            'я': [0],
            'в': [],
            'он': [0],
            'майка': [1],
            'соломка': [3],
            'изжить': [3],
            'виться': [1],
            'данный': [1],
            'зорька': [1],
            'банка': [1],
            'оттечь': [3],
            'советского': [3],
            'союза': [2],
            'изжила': [5],
            'автоподъёмник': [8],
            'каракуля': [3],
            'супервайзер': [6],
            'колесом': [5]
        }
        for word, pos in checks.items():
            predicted = self.model.predict(word, schema=PredictSchema.CLASSIC)
            predicted = list(sorted(predicted))
            target = list(sorted(pos))
            self.assertEqual(predicted, target, msg="{}: {} vs {}".format(word, predicted, target))
