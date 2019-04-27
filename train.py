import os
import argparse

from allennlp.data.vocabulary import Vocabulary
from allennlp.common.params import Params
from allennlp.models import Model

from russ.stress.reader import StressReader
from russ.stress.model import StressModel


def train(model_path, train_path, val_path=None, vocabulary_path=None, config_path=None):
    config_path = config_path or os.path.join(model_path, "config.json")
    params = Params.from_file(config_path)
    print(train_path)

    vocabulary_path = vocabulary_path or os.path.join(model_path, "vocabulary")
    if not os.path.isdir(vocabulary_path):
        reader = StressReader()
        train_dataset = reader.read(train_path)
        vocab_params = params.pop("vocabulary", Params({}))
        vocabulary = Vocabulary.from_params(vocab_params, instances=train_dataset)
        vocabulary.save_to_files(vocabulary_path)
    else:
        vocabulary = Vocabulary.from_files(vocabulary_path)

    print("Vocabulary ok")

    train_params = params.pop("train", Params({}))
    model = StressModel.from_params(params, vocab=vocabulary)
    model.train_file(train_path, train_params, serialization_dir=model_path, valid_file_name=val_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--val-path')
    parser.add_argument('--vocabulary-path')
    parser.add_argument('--config-path')
    args = parser.parse_args()
    train(**vars(args))
