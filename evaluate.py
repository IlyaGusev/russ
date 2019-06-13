import os
import argparse
import torch

from russ.stress.model import StressModel


def evaluate(model_path, test_path, config_path):
    params_path = config_path or os.path.join(model_path, "config.json")
    device = 0 if torch.cuda.is_available() else -1
    model = StressModel.load(model_path, params_path, cuda_device=device)

    correct_count = 0
    all_count = 0
    errors = []
    with open(test_path, "r", encoding="utf-8") as r:
        for word in r:
            word = word.strip()
            predicted_stresses = model.predict_word_stress(word)
            target_stresses = [i-1 for i, tag in enumerate(model.reader.text_to_instance(word)["tags"]) if tag == '1']
            is_ok_word = len(set(predicted_stresses).intersection(set(target_stresses))) != 0
            correct_count += int(is_ok_word)
            all_count += 1
            if not is_ok_word:
                errors.append((word, predicted_stresses, target_stresses))
    print("WER: {0:.2f}".format(100.*(1.0 - correct_count/float(all_count))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--config-path', default=None)
    args = parser.parse_args()
    evaluate(**vars(args))
