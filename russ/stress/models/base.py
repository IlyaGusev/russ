from typing import Dict
import torch
from allennlp.data.dataset_readers import DatasetReader
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
                 reader: DatasetReader = None,
                 embedder: TextFieldEmbedder = None,
                 encoder: Seq2SeqEncoder = None,
                 dropout: float = 0.0) -> None:
        super().__init__(vocab)
        self.reader = reader
        self.embedder = embedder
        self.embeddings_dropout = torch.nn.Dropout(p=dropout)
        self.encoder = encoder
        self.encoder_dropout = torch.nn.Dropout(p=dropout)
        vocab_size = vocab.get_vocab_size('labels')
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(), out_features=vocab_size)
        self.accuracy = CategoricalAccuracy()
        self.awer = BooleanAccuracy()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                tags: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.embedder.forward(tokens)
        embeddings = self.embeddings_dropout(embeddings)
        encoder_out = self.encoder.forward(embeddings, mask)
        encoder_out = self.encoder_dropout(encoder_out)
        logits = self.hidden2tag(encoder_out)
        output = {"logits": logits}
        if tags is not None:
            self.accuracy(logits, tags, mask)
            self.awer(logits.max(dim=2)[1], tags, mask)
            output["loss"] = sequence_cross_entropy_with_logits(logits, tags, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset), "awer": self.awer.get_metric(reset)}