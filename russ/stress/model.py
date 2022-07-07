from enum import Enum

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


class PredictSchema(Enum):
    CLASSIC = 0
    CONSTRAINED = 1


class StressTransformerModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision="3400828"
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def predict_batch(self, texts, schema: PredictSchema = PredictSchema.CLASSIC):
        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.model(**inputs)
        logits = outputs.logits
        batch_scores = torch.nn.functional.softmax(logits, dim=2)

        predicted_stresses = []
        for text, scores in zip(texts, batch_scores):
            scores = scores[1:len(text) + 1]

            stresses = []
            if schema == PredictSchema.CLASSIC:
                classes = scores.argmax(dim=1).tolist()
                assert len(classes) == len(text)
                stresses = [i for i, stress_type in enumerate(classes) if stress_type == 1]

            elif schema == PredictSchema.CONSTRAINED:
                stress = scores[:, 1].argmax().item()
                stresses = [stress]

            if not stresses:
                primary_stress_index = scores[:, 0].argmin()
                stresses = [primary_stress_index]
            predicted_stresses.append(stresses)
        return predicted_stresses

