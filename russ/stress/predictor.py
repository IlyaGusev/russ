import os
import logging
from typing import List
from collections import OrderedDict

from torch.nn.functional import softmax
from torch import Tensor

from russ.syllables import get_first_vowel_position, get_syllables
from russ.stress.dict import StressDict, Stress
from russ.stress.model import StressModel, PredictSchema

logger = logging.getLogger(__name__)


class StressPredictor:
    def __init__(
        self,
        model_name: str = "IlyaGusev/ru-word-stress-transformer",
        revision: str = "b44e211",
        device: str = "cpu",
        stress_dict_path: str = None,
        cache_capacity: int = 100000
    ):
        self.model = StressModel(model_name, device=device, revision=revision)
        self.stress_dict = None
        self.cache = OrderedDict()
        self.cache_capacity = cache_capacity
        if stress_dict_path:
            self.stress_dict = StressDict()
            self.stress_dict.load(stress_dict_path)

    def predict_words(
        self,
        words: List[str],
        schema: PredictSchema = PredictSchema.CONSTRAINED,
        batch_size: int = 2048
    ) -> List[List[int]]:
        stresses = {}
        for word in words:
            syllables = get_syllables(word)
            if len(syllables) <= 1:
                first_vowel_pos = get_first_vowel_position(word)
                stresses[word] = [] if first_vowel_pos == -1 else [first_vowel_pos]
                continue

            if self.stress_dict is not None:
                dict_record = self.stress_dict.get(word, Stress.Type.PRIMARY)
                stresses[word] = dict_record

            if word in self.cache:
                self.cache.move_to_end(word)
                stresses[word] = self.cache.get(word)

        unk_words = [word for word in words if word not in stresses]
        if not unk_words:
            return [stresses[word] for word in words]
        results = self.model.predict(unk_words, schema, batch_size=batch_size)
        for word, word_stresses in zip(unk_words, results):
            self.cache[word] = word_stresses
            stresses[word] = word_stresses
            if len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)
        return [stresses[word] for word in words]

    def predict(self, word: str, schema: PredictSchema = PredictSchema.CONSTRAINED):
        return self.predict_words([word], schema)[0]

    def __str__(self):
        s = str(self.model) + "\n"
        s += "Model params count: {}\n".format(sum(p.numel() for p in self.model.parameters()))
        if self.stress_dict is None:
            s += "No stress dictionary"
        else:
            s += "{} words in stress dictionary".format(len(self.stress_dict))
        return s
