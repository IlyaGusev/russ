import os
import logging
from typing import List

from torch.nn.functional import softmax
from torch import Tensor

from russ.syllables import get_first_vowel_position, get_syllables
from russ.stress.dict import StressDict, Stress
from russ.stress.model import StressTransformerModel, PredictSchema

logger = logging.getLogger(__name__)


class StressPredictor:
    def __init__(
        self,
        model_name: str,
        stress_dict_path: str = None
    ):
        super().__init__()

        self.model = StressTransformerModel(model_name)
        self.stress_dict = StressDict()
        if stress_dict_path:
            self.stress_dict.load(stress_dict_path)

    def predict_words_stresses(self, words: List[str], schema: PredictSchema = PredictSchema.CONSTRAINED):
        stresses = {}
        for word in words:
            syllables = get_syllables(word)
            if len(syllables) <= 1:
                first_vowel_pos = get_first_vowel_position(word)
                stresses[word] = [] if first_vowel_pos == -1 else [first_vowel_pos]
                continue

            if len(self.stress_dict) != 0:
                dict_record = self.stress_dict.get(word, Stress.Type.PRIMARY)
                stresses[word] = dict_record

        batch = [word for word in words if word not in stresses]
        if not batch:
            return stresses
        results = self.model.predict_batch(batch, schema)
        stresses = {word: word_stresses for word, word_stresses in zip(batch, results)}
        return stresses

    def predict(self, word: str, schema: PredictSchema = PredictSchema.CONSTRAINED):
        return self.predict_words_stresses([word], schema)[word]

    def __str__(self):
        s = str(self.model) + "\n"
        s += "Model params count: {}\n".format(sum(p.numel() for p in self.model.parameters()))
        if self.stress_dict is None:
            s += "No stress dictionary"
        else:
            s += "{} words in stress dictionary".format(len(self.stress_dict))
        return s
