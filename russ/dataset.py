import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from russ.tokenizer import CharTokenizer
from russ.convert import convert_to_record


class StressDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: CharTokenizer,
        max_length: int = 40,
        sample_rate: float = 1.0,
        skip_secondary: bool = False,
        convert_secondary: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.skip_secondary = skip_secondary
        self.convert_secondary = convert_secondary

        self.records = []
        with open(file_path) as r:
            for line in tqdm(r):
                if random.random() > sample_rate:
                    continue
                record = self.convert(line)
                self.records.append(record)

    def convert(self, text):
        record = convert_to_record(
            text,
            skip_secondary=self.skip_secondary,
            convert_secondary=self.convert_secondary
        )
        inputs = self.tokenizer(
            record["text"],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        labels_tensor = input_ids.new_full(input_ids.size(), -100)
        max_length = len(input_ids)
        tags = record["tags"][:max_length - 2]
        has_primary = bool([i for i in tags if i == 1])
        assert has_primary

        for i, tag in enumerate(tags):
            labels_tensor[i + 1] = tag
        inputs["labels"] = labels_tensor

        return inputs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
