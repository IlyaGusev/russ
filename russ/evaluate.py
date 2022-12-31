import argparse
import time

import torch
from transformers import AutoModelForTokenClassification, pipeline

from russ.stress.predictor import StressPredictor
from russ.convert import convert_to_record


def evaluate(input_path, model_path, batch_size):
    records = []
    with open(input_path) as r:
        for i, line in enumerate(r):
            record = convert_to_record(line)
            records.append(record)

    correct_cnt, all_cnt = 0, 0
    texts = [r["text"] for r in records]
    labels = [r["tags"] for r in records]

    model = StressPredictor(model_path)
    start_time = time.perf_counter_ns()
    predictions = model.predict_words(texts, batch_size=batch_size)
    assert len(predictions) == len(labels) == len(texts), f"{len(predictions)} vs {len(labels)}"
    end_time = time.perf_counter_ns()
    print("CPU time, micros, per sample:", (end_time - start_time) // 1000 // len(predictions))

    model = StressPredictor(model_path, device="cuda")
    start_time = time.perf_counter_ns()
    predictions = model.predict_words(texts, batch_size=batch_size)
    assert len(predictions) == len(labels) == len(texts), f"{len(predictions)} vs {len(labels)}"
    end_time = time.perf_counter_ns()
    print("GPU time, micros, per sample:", (end_time - start_time) // 1000 // len(predictions))

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
