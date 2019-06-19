import argparse

from russ.stress.dict import StressDict, Stress
from russ.stress.reader import StressReader


def create_dict(files, output):
    stress_dict = StressDict()
    reader = StressReader()
    for path in files:
        with open(path, "r", encoding="utf-8") as r:
            for line in r:
                word = line.strip()
                schema = list(map(int, reader.text_to_instance(word)["tags"].labels))
                primary_stresses = [Stress(i-1, Stress.Type.PRIMARY) for i, stress in enumerate(schema) if stress == 1]
                secondary_stresses = [Stress(i-1, Stress.Type.SECONDARY) for i, stress in enumerate(schema) if stress == 2]
                clean_word = word.replace("`", "").replace("'", "")
                stress_dict.update(clean_word, primary_stresses + secondary_stresses)
    stress_dict.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files',  nargs='*')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    create_dict(**vars(args))
