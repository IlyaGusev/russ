from typing import Dict

from torch import Tensor
from torch.nn import Dropout, Linear, ReLU
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from allennlp.models import Model


@Model.register("base")
class BaseModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder = None,
                 encoder: Seq2SeqEncoder = None,
                 embeddings_dropout: float = 0.0,
                 encoder_dropout: float = 0.0,
                 dense_dim: int = None) -> None:
        super().__init__(vocab)

        self._vocab_size = vocab.get_vocab_size('labels')
        self._dense_dim = dense_dim

        self._embedder = embedder
        self._embeddings_dropout = Dropout(p=embeddings_dropout)
        self._encoder = encoder
        self._encoder_dropout = Dropout(p=encoder_dropout)
        self._encoder_output_dim = encoder.get_output_dim()
        self._hidden_input_dim = self._encoder_output_dim
        if dense_dim:
            self._dense = Linear(self._encoder_output_dim, dense_dim)
            self._dense_relu = ReLU()
            self._hidden_input_dim = dense_dim
        self._hidden2tag = Linear(self._hidden_input_dim, self._vocab_size)
        self._accuracy = CategoricalAccuracy()
        self._boolean_accuracy = BooleanAccuracy()

    def forward(self,
                tokens: Dict[str, Tensor],
                tags: Tensor = None) -> Dict[str, Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self._embedder.forward(tokens)
        embeddings = self._embeddings_dropout.forward(embeddings)
        encoder_out = self._encoder.forward(embeddings, mask)
        encoder_out = self._encoder_dropout.forward(encoder_out)
        if self._dense_dim:
            encoder_out = self._dense_relu(self._dense(encoder_out))

        logits = self._hidden2tag(encoder_out)

        output = {"logits": logits}
        if tags is not None:
            self._accuracy(logits, tags, mask)
            self._boolean_accuracy(logits.max(dim=2)[1], tags, mask)
            output["loss"] = sequence_cross_entropy_with_logits(logits, tags, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset), "wer": 1. - self._boolean_accuracy.get_metric(reset)}