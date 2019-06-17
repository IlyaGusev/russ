import os
import argparse
import torch

from russ.stress.model import StressModel


class WordErrorRate:
    def __init__(self):
        self.correct = 0
        self.all = 0

    def add(self, is_ok: bool):
        self.all += 1
        if is_ok:
            self.correct += 1

    def __str__(self):
        return "WER: {0:.2f}".format(100.*(1.0 - self.correct/float(self.all)))


def evaluate(model_path, test_path, config_path, errors_file_path,
             metric, max_count, report_every, batch_size):
    params_path = config_path or os.path.join(model_path, "config.json")
    device = 0 if torch.cuda.is_available() else -1
    model = StressModel.load(model_path, params_path, cuda_device=device)

    wer = WordErrorRate()
    errors = []
    batch = []
    modes = {
        "wer": StressModel.PredictSchema.CLASSIC,
        "wer-constrained": StressModel.PredictSchema.CONSTRAINED
    }
    mode = modes[metric]
    with open(test_path, "r", encoding="utf-8") as r:
        for line_num, word in enumerate(r):
            if report_every and line_num != 0 and line_num % report_every == 0:
                print(line_num, wer)
            if max_count and line_num >= max_count:
                break
            word = word.strip()
            batch.append(word)
            if len(batch) < batch_size:
                continue
            results = model.predict_words_stresses(batch, mode)
            for word, predicted_stresses in results.items():
                target_stresses = [i-1 for i, tag in enumerate(model.reader.text_to_instance(word)["tags"]) if tag == '1']
                is_ok_word = set(predicted_stresses) <= set(target_stresses)
                if len(target_stresses) != 0 and len(predicted_stresses) == 0:
                    is_ok_word = False
                wer.add(is_ok_word)
                if not is_ok_word:
                    errors.append((word, predicted_stresses, target_stresses))
            batch = []
    print(wer)
    if errors_file_path:
        with open(errors_file_path, "w", encoding="utf-8") as w:
            w.write("word\tpredicted\ttarget\n")
            for word, predicted, target in errors:
                w.write("{}\t{}\t{}\n".format(word, ";".join(map(str, predicted)), ";".join(map(str, target))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--errors-file-path', default=None)
    parser.add_argument('--metric', default="wer-constrained", choices=("wer", "wer-constrained"))
    parser.add_argument('--max-count', default=None)
    parser.add_argument('--report-every', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    evaluate(**vars(args))
