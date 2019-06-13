# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты предсказателя ударений.

import unittest
import os

from allennlp.common.params import Params
from allennlp.models import Model

from russ.stress.model import StressModel


class TestStressPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = "/media/yallen/My Passport/Projects/russ/models/big"
        cls.model = StressModel.load(model_path)

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
            # 'пора': [3, 1],
            'изжила': [5],
            'меда': [1],
            'автоподъёмник': [8],
            'каракуля': [3],
            'супервайзер': [6],
            'колесом': [5]
        }
        for word, pos in checks.items():
            predicted = list(sorted(self.model.predict_word_stress(word)))
            target = list(sorted(pos))
            self.assertEqual(predicted, target, msg="{}: {} vs {}".format(word, predicted, target))
