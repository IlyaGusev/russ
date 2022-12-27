from enum import Enum
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from russ.lstm import LstmModelForTokenClassification
from russ.tokenizer import CharTokenizer


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


class PredictSchema(Enum):
    CLASSIC = 0
    CONSTRAINED = 1


class StressModel:
    def __init__(self, model_name, lower: bool = False, max_length: int = 40):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.lower = lower
        self.max_length = max_length

    def preprocess_text(self, text):
        text = text[:self.max_length-2]
        return text

    def predict_batch(self, texts, schema: PredictSchema):
        texts = [self.preprocess_text(text) for text in texts]
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.model(**inputs)
        logits = outputs["logits"]
        batch_scores = torch.nn.functional.softmax(logits, dim=2)

        predicted_stresses = []
        for text, scores in zip(texts, batch_scores):
            scores = scores[1:len(text) + 1]

            stresses = []
            if schema == PredictSchema.CLASSIC:
                classes = scores.argmax(dim=1).tolist()
                assert len(classes) == len(text), f"{classes}, {text}"
                stresses = [i for i, stress_type in enumerate(classes) if stress_type == 1]

            elif schema == PredictSchema.CONSTRAINED:
                stress = scores[:, 1].argmax().item()
                stresses = [stress]

            if not stresses:
                primary_stress_index = scores[:, 0].argmin()
                stresses = [primary_stress_index]
            predicted_stresses.append(stresses)
        return predicted_stresses

    def predict(self, texts: List[str], schema: PredictSchema = PredictSchema.CLASSIC, batch_size: int = 2048):
        predicted_stresses = []
        for batch_texts in gen_batch(texts, batch_size):
            batch_predicted_stresses = self.predict_batch(batch_texts, schema)
            predicted_stresses.extend(batch_predicted_stresses)
        return batch_predicted_stresses

