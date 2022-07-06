import argparse
from tqdm import tqdm

import torch
from transformers import AutoModelForTokenClassification, pipeline

from russ.tokenizer import CharTokenizer
from russ.convert import convert_to_record
from russ.dataset import StressDataset


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def predict_batch(texts, tokenizer, model, max_length):
    inputs = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    outputs = model(**inputs)
    logits = outputs.logits
    batch_scores = torch.nn.functional.softmax(logits, dim=2)
    batch_predicted_labels = []
    for text, scores in zip(texts, batch_scores):
        ps = scores[1:len(text) + 1, 1]
        ss = scores[1:len(text) + 1, 2]
        primary_stress_index = ps.argmax()
        secondary_stress_index = ss.argmax()
        primary_stress_score = ps[primary_stress_index]
        secondary_stress_score = ss[secondary_stress_index]

        predicted_labels = [0 for _ in range(len(text))]
        predicted_labels[primary_stress_index] = 1
        if secondary_stress_score > 0.5:
            predicted_labels[secondary_stress_index] = 2
        batch_predicted_labels.append(predicted_labels)
    return batch_predicted_labels


def evaluate(input_path, model_path, batch_size):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = CharTokenizer.from_pretrained(model_path)

    records = []
    with open(input_path) as r:
        for i, line in tqdm(enumerate(r)):
            if i == 10000:
                break
            record = convert_to_record(line)
            records.append(record)

    correct_cnt, all_cnt = 0, 0
    for batch in gen_batch(records, batch_size):
        batch_texts = [r["text"] for r in batch]
        batch_labels = [r["tags"] for r in batch]
        batch_predicted_labels = predict_batch(batch_texts, tokenizer, model, max_length=40)
        for text, true_labels, pred_labels in zip(batch_texts, batch_labels, batch_predicted_labels):
            pred_labels = [i if i != 2 else 0 for i in pred_labels]
            true_labels = [i if i != 2 else 0 for i in true_labels]
            is_correct = True
            for pred_tag, true_tag in zip(pred_labels, true_labels):
                if pred_tag == 1 and true_tag != 1:
                    is_correct = False
            correct_cnt += int(is_correct)
            all_cnt += 1
            if not is_correct:
                print(text)
                print(true_labels)
                print(pred_labels)
    print(correct_cnt, all_cnt, 100.0 * correct_cnt / all_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    evaluate(**vars(args))
