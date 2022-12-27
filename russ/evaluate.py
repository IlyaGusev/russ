import argparse

import torch
from transformers import AutoModelForTokenClassification, pipeline

from russ.stress.model import StressModel
from russ.convert import convert_to_record


def evaluate(input_path, model_path, batch_size):
    model = StressModel(model_path)

    records = []
    with open(input_path) as r:
        for i, line in enumerate(r):
            record = convert_to_record(line)
            records.append(record)

    correct_cnt, all_cnt = 0, 0
    texts = [r["text"] for r in records]
    labels = [r["tags"] for r in records]
    predictions = model.predict(texts, batch_size=batch_size)
    for text, true_labels, pred_stresses in zip(texts, labels, predictions):
        for index in pred_stresses:
            assert index < len(true_labels), f"{true_labels}, {pred_stresses}, {text}"
        is_correct = all([true_labels[index] == 1 for index in pred_stresses])
        correct_cnt += int(is_correct)
        all_cnt += 1
    print(correct_cnt, all_cnt, 100.0 * correct_cnt / all_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    evaluate(**vars(args))
