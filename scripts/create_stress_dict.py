import argparse

from russ.stress.dict import StressDict, Stress
from russ.convert import convert_to_record


def create_dict(files, output):
    stress_dict = StressDict()
    for path in files:
        with open(path, "r", encoding="utf-8") as r:
            for line in r:
                word = line.strip()
                record = convert_to_record(word)
                schema = record["tags"]
                clean_word = record["text"]
                primary_stresses = [
                    Stress(i - 1, Stress.Type.PRIMARY)
                    for i, stress in enumerate(schema) if stress == 1
                ]
                secondary_stresses = [
                    Stress(i - 1, Stress.Type.SECONDARY)
                    for i, stress in enumerate(schema) if stress == 2
                ]
                stress_dict.update(clean_word, primary_stresses + secondary_stresses)
    stress_dict.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    create_dict(**vars(args))
