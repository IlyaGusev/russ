import sys
import json
from lxml import etree

from tqdm import tqdm

from russ.syllables import get_syllables


def is_good_sentence(sentence):
    if len(sentence) > 200:
        return False
    if len(sentence) < 10:
        return False
    return True


def write_jsonl(records, path):
    with open(path, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


def main(
    input_file,
    output_file
):
    root = etree.parse(input_file)
    sentences_nodes = root.xpath('//sentence')

    records = []
    for sentence_node in tqdm(sentences_nodes):
        sentence_id = sentence_node.xpath('./@id')[0]
        sentence = sentence_node.xpath('.//source/text()')[0]
        tokens_nodes = sentence_node.xpath('.//token')
        if not is_good_sentence(sentence):
            continue
        tokens = sentence_node.xpath('.//token/@text')
        for token_node in tokens_nodes:
            token_id = token_node.xpath('./@id')[0]
            token = token_node.xpath('./@text')[0]
            gram_value = "|".join(token_node.xpath('.//g/@v'))
            syllables = get_syllables(token)
            if len(syllables) <= 1:
                continue
            options = []
            for syllable in syllables:
                vowel_pos = syllable.vowel()
                option = token.lower()
                vowel = option[vowel_pos]
                option = option[:vowel_pos] + vowel.upper() + chr(39) + option[vowel_pos+1:]
                options.append(option)
            records.append({
                "sentence_id": sentence_id,
                "token_id": token_id,
                "sentence": sentence,
                "tokens": tokens,
                "token": token,
                "gram_value": gram_value,
                "stress_options": options
            })
    write_jsonl(records, output_file)


input_file = sys.argv[1]
output_file = sys.argv[2]
main(input_file, output_file)
