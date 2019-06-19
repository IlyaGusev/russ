import os
import logging
from enum import Enum
from typing import List

from torch.nn.functional import softmax
from torch import Tensor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.data.iterators import DataIterator
from allennlp.training.trainer import Trainer

from russ.stress.predictor import StressPredictor
from russ.syllables import get_first_vowel_position, get_syllables
from russ.settings import RU_MAIN_MODEL
from russ.stress.dict import StressDict, Stress

logger = logging.getLogger(__name__)


class StressModel(Registrable):

    class PredictSchema(Enum):
        CLASSIC = 0
        CONSTRAINED = 1

    def __init__(self,
                 model: Model,
                 vocab: Vocabulary,
                 reader: DatasetReader = None,
                 stress_dict: StressDict = None):
        super().__init__()
        self.reader = reader
        self.vocab = vocab
        self.model = model
        self.stress_dict = stress_dict

    def train_file(self,
                   train_file_name: str,
                   train_params: Params,
                   serialization_dir: str = None,
                   valid_file_name: str = None):
        assert os.path.exists(train_file_name)
        assert not valid_file_name or os.path.exists(valid_file_name)
        train_dataset = self.reader.read(train_file_name)
        valid_dataset = self.reader.read(valid_file_name) if valid_file_name else None

        iterator = DataIterator.from_params(train_params.pop('iterator'))
        iterator.index_with(self.vocab)
        trainer = Trainer.from_params(self.model, serialization_dir, iterator,
                                      train_dataset, valid_dataset, train_params.pop('trainer'))
        train_params.assert_empty("Trainer")
        return trainer.train()

    def predict_words_stresses(self, words: List[str], schema: PredictSchema = PredictSchema.CONSTRAINED):
        stresses = {}
        for word in words:
            syllables = get_syllables(word)
            if len(syllables) <= 1:
                stresses[word] = [] if get_first_vowel_position(word) == -1 else [get_first_vowel_position(word)]
                continue

            dict_record = self.stress_dict.get(word, Stress.Type.PRIMARY) if self.stress_dict is not None else None
            if dict_record is not None:
                stresses[word] = dict_record

        batch = [{"word": word} for word in words if word not in stresses]
        if not batch:
            return stresses
        self.model.eval()
        predictor = StressPredictor(self.model, dataset_reader=self.reader)
        results = predictor.predict_batch_json(batch)
        for word, result in zip(batch, results):
            word = word["word"]
            logits = result['logits']
            probabilities = softmax(Tensor(logits), dim=1)[1:len(word)+1]
            if schema == StressModel.PredictSchema.CLASSIC:
                classes = probabilities.max(dim=1)[1].cpu().tolist()
                stresses[word] = [i for i, stress_type in enumerate(classes) if stress_type == 1]
            elif schema == StressModel.PredictSchema.CONSTRAINED:
                stress = probabilities[:, 1].max(dim=0)[1].cpu().item()
                stresses[word] = [stress]
        return stresses

    def predict(self, word: str, schema: PredictSchema = PredictSchema.CONSTRAINED):
        return self.predict_words_stresses([word], schema)[word]

    @classmethod
    def load(cls,
             serialization_dir: str = RU_MAIN_MODEL,
             params_file: str = None,
             weights_file: str = None,
             vocabulary_dir: str = None,
             stress_dict_path: str = None,
             cuda_device: int = -1,
             **kwargs) -> 'StressModel':
        params_file = params_file or os.path.join(serialization_dir, "config.json")
        params = Params.from_file(params_file)
        params.pop("vocab", None)

        vocabulary_dir = vocabulary_dir or os.path.join(serialization_dir, "vocabulary")
        vocab = Vocabulary.from_files(vocabulary_dir)

        if params.get('train', None):
            params.pop('train')

        inner_model = Model.load(
            params,
            serialization_dir,
            weights_file=weights_file,
            cuda_device=cuda_device)
        params.pop('model')
        if params.get('vocabulary', None):
            params.pop('vocabulary')

        stress_dict = None
        if stress_dict_path:
            stress_dict = StressDict()
            stress_dict.load(stress_dict_path)

        model = StressModel.from_params(params, model=inner_model, vocab=vocab, stress_dict=stress_dict, **kwargs)

        return model

    def __str__(self):
        s = str(self.model) + "\n"
        s += "Trainable params count: {}\n".format(sum(p.numel() for p in self.model.parameters()if p.requires_grad))
        if self.stress_dict is None:
            s += "No stress dictionary"
        else:
            s += "{} words in stress dictionary".format(len(self.stress_dict))
        return s
