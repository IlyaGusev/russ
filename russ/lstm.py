import copy

import torch
from torch.nn.functional import pad
from torch.nn import CrossEntropyLoss, Embedding, Dropout, LSTM, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

class LstmModelConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(
        self,
        vocab_size=70,
        hidden_size=256,
        num_hidden_layers=2,
        id2label=dict(),
        label2id=dict(),
        max_length=40,
        num_labels=3,
        pad_token_id=0,
        dropout_rate: float = 0.4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.id2label = id2label
        self.label2id = label2id
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.num_hidden_layers = num_hidden_layers

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class LstmModelForTokenClassification(PreTrainedModel):
    config_class = LstmModelConfig

    def __init__(self, config=None):
        super().__init__(config)

        self.embeddings_layer = Embedding(config.vocab_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_rate)
        self.lstm_layer = LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=config.num_hidden_layers
        )
        self.out_layer = Linear(config.hidden_size, config.num_labels)
        self.loss_fct = CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        #padded_input_ids = pack_padded_sequence(input_ids, attention_mask.sum(dim=-1), batch_first=True)
        projections = self.embeddings_layer.forward(input_ids)
        projections = projections.reshape(projections.size(0), projections.size(1), -1)
        output, _= self.lstm_layer(projections)
        #output, input_sizes = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        output = self.out_layer.forward(output)
        result = {"logits": output}
        if labels is not None:
            output = output.transpose(1, 2)
            result["loss"] = self.loss_fct(output, labels)
        return result


AutoConfig.register("lstm", LstmModelConfig)
AutoModel.register(LstmModelConfig, LstmModelForTokenClassification)
AutoModelForTokenClassification.register(LstmModelConfig, LstmModelForTokenClassification)
