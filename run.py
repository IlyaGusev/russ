import argparse

from russ.stress.model import StressModel


def run(word, words_file_path, batch_size, text_file_path, model_path):
    model = StressModel.load() if not model_path else StressModel.load(model_path)
    if word:
        print(model.predict(word))
    elif words_file_path:
        with open(words_file_path, "r", encoding="utf-8") as r:
            batch = []
            for line in r:
                word = line.strip()
                batch.append(word)
                if len(batch) == batch_size:
                    stresses = model.predict_words_stresses(batch)
                    for word, stress in stresses.items():
                        print("{}\t{}".format(word, ",".join(map(str, stress))))
                    batch = []
            if batch:
                stresses = model.predict_words_stresses(batch)
                for word, stress in stresses.items():
                    print("{}\t{}".format(word, ",".join(map(str, stress))))
    elif text_file_path:
        with open(text_file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = line.strip()
                words = line.split(" ")
                for word_num, word in enumerate(words):
                    stresses = model.predict(word)
                    for i, stress in enumerate(stresses):
                        word = word[:stress+i+1] + chr(39)  + word[stress+i+1:]
                    words[word_num] = word
                print(" ".join(words))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', default=None)
    parser.add_argument('--text-file-path', default=None)
    parser.add_argument('--words-file-path', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-path', default=None)
    args = parser.parse_args()
    run(**vars(args))
