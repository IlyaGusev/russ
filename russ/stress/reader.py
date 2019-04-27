from typing import Dict, List, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import CharacterTokenizer


@DatasetReader.register("stress")
class StressReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,) -> None:
        super().__init__(lazy=True)

        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path: str):
        for word in self._lines(file_path):
            yield self.text_to_instance(word)

    def text_to_instance(self, text: str) -> Instance:
        clean_word = ""
        schema = []
        for ch in text:
            if ch == "'":
                schema[-1] = 1
            elif ch == "`":
                schema[-1] = 2
            elif ch == "ё":
                schema.append(1)
                clean_word += ch
            else:
                schema.append(0)
                clean_word += ch
        assert len(schema) == len(clean_word)
        tokens = self._tokenize(clean_word)
        if not schema:
            schema = None
        else:
            schema.insert(0, 0)
            schema.append(0)
        return self._sample_to_instance(tokens, schema)

    def _tokenize(self, text: str) -> List[Token]:
        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text.insert(0, Token(START_SYMBOL))
        tokenized_text.append(Token(END_SYMBOL))
        return tokenized_text

    def _sample_to_instance(self, sample: List[Token], schema: List[int]=None) -> Instance:
        result = dict()
        result['tokens'] = TextField(sample, self._token_indexers)
        if schema:
            result['tags'] = SequenceLabelField(labels=list(map(str, schema)), sequence_field=result['tokens'])
        return Instance(result)

    @staticmethod
    def _lines(file_path: str) -> Iterable[str]:
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            for line in text_file:
                line = line.strip()
                yield line
