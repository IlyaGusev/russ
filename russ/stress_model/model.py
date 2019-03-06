from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from russ.stress_model.stress_reader import StressReader


class LSTMTagger(Model):
    def __init__(self,
                 embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embeddings = embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.embeddings.forward(tokens)
        encoder_out = self.encoder.forward(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def train():
    reader = StressReader()
    from russ.settings import RU_ALL_DICT
    train_dataset = reader.read(RU_ALL_DICT)
    vocab = Vocabulary.from_instances(train_dataset)
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True))
    model = LSTMTagger(word_embeddings, lstm, vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    iterator = BucketIterator(batch_size=64, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=None,
                      patience=2,
                      num_epochs=100)
    trainer.train()
    # predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    # tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    # tag_ids = np.argmax(tag_logits, axis=-1)
    # print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
    # # Here's how to save the model.
    # with open("/tmp/model.th", 'wb') as f:
    #     torch.save(model.state_dict(), f)
    # vocab.save_to_files("/tmp/vocabulary")
    # # And here's how to reload the model.
    # vocab2 = Vocabulary.from_files("/tmp/vocabulary")
    # model2 = LstmTagger(word_embeddings, lstm, vocab2)
    # with open("/tmp/model.th", 'rb') as f:
    #     model2.load_state_dict(torch.load(f))
    # predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    # tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
    # assert tag_logits2 == tag_logits

train()