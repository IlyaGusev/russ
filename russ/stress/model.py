import os
import logging
from enum import Enum

import torch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.data.iterators import DataIterator
from allennlp.training.trainer import Trainer

from russ.stress.reader import StressReader
from russ.stress.predictor import StressPredictor
from russ.syllables import get_first_vowel_position, get_syllables

logger = logging.getLogger(__name__)


class StressModel(Registrable):

    class PredictSchema(Enum):
        LOCAL = 0
        GLOBAL = 1

    def __init__(self,
                 model: Model,
                 vocab: Vocabulary,
                 reader: DatasetReader = None) -> None:
        super().__init__()
        self.reader = reader
        self.vocab = vocab
        self.model = model

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

    def predict_word_stress(self, word: str, schema: PredictSchema = PredictSchema.GLOBAL):
        self.model.eval()
        syllables = get_syllables(word)
        if len(syllables) <= 1:
            return [] if get_first_vowel_position(word) == -1 else [get_first_vowel_position(word)]
        predictor = StressPredictor(self.model, dataset_reader=self.reader)
        logits = predictor.predict(word)['logits']
        probabilities = torch.nn.Softmax(dim=1)(torch.Tensor(logits))[1:-1]
        if schema == StressModel.PredictSchema.LOCAL:
            stresses = probabilities.max(dim=1)[1].cpu().tolist()
            return [i for i, stress_type in enumerate(stresses) if stress_type == 1]
        elif schema == StressModel.PredictSchema.GLOBAL:
            stress = probabilities[:, 1].max(dim=0)[1].cpu().item()
            return [stress]

    def log_model(self):
        logger.info(self.model)
        params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Trainable params count: " + str(params_count))

    @classmethod
    def load(cls,
             serialization_dir: str,
             params_file: str = None,
             weights_file: str = None,
             vocabulary_dir: str = None,
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
        model = StressModel.from_params(params, model=inner_model, vocab=vocab, **kwargs)

        return model
