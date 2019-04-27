from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance


@Predictor.register('stress')
class StressPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, word: str) -> JsonDict:
        return self.predict_json({"word": word})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["word"])
