# Python library for word stress detection #

[![Tests Status](https://github.com/IlyaGusev/russ/actions/workflows/python-package.yml/badge.svg)](https://github.com/IlyaGusev/russ/actions/workflows/python-package.yml)
[![PyPI Version](https://img.shields.io/pypi/v/russ.svg)](https://pypi.python.org/pypi/russ)
[![codecov](https://codecov.io/gh/IlyaGusev/russ/branch/master/graph/badge.svg)](https://codecov.io/gh/IlyaGusev/russ)

### Install
```
pip install russ
```

or

```
git clone https://github.com/IlyaGusev/russ
cd russ
pip install -r requirements.txt
python setup.py
```

### Usage

Colab: [link](https://colab.research.google.com/drive/1rv_NkyI7-EY45XZb0Bax2xR2Ms71C1lc)

```
from russ.stress.predictor import StressPredictor

model = StressPredictor()
model.predict("корова")

>>> [3]
```

### Dataset
* Train/val/test split: [link](https://www.dropbox.com/s/0c4xesynenj1xyx/russ_dataset.tar.gz)

### Metrics
* CPU, AMD EPYC 7282, batch size = 2048: 1495 μs for 1 word
* GPU, 1x RTX 3090, batch size = 2048: 173 μs for 1 word
* Test accuracy: 89.73%
* All accuracy: 97.95%

### Commands

#### download.sh

Script for downloading datasets:
* ru_custom.txt: 885 words
* zaliznyak.txt: 86839 lexemes
* espeak.txt: 804909 words
* ruwiktionary-20221201-pages-articles.xml: articles from ruwiktionary, update to a new dump

#### scripts/prepare_data.py

Preparing data for training

| Argument               | Default | Description                                                                         |
|:-----------------------|:--------|:------------------------------------------------------------------------------------|
| --wiktionary-dump-path | None    | path to downloaded wiktionary dump                                                  |
| --espeak-dump-path     | None    | path to espeak dump                                                                 |
| --custom-dict-path     | None    | path to file with custom words                                                      |
| --inflected-dict-path  | None    | path to downloaded file with lexemes                                                |
| --inflected-sample-rate | 0.3    | part of inflected dict to use                                                       |
| --split-mode           | lexemes | how to split into train, val and test files: "sort", "lexemes" or "shuffle"         |
| --all-path           | data/all.txt        | path to output train file                                                           |
| --train-path           | data/train.txt | path to output train file                                                           |
| --val-path             | data/val.txt | path to output validation file                                                      |
| --test-path            | data/test.txt | path to output test file                                                            |
| --val-part             | 0.05    | part of validation file                                                             |
| --test-part            | 0.05    | part of test file                                                                   |
| --lower                | Fasle   | lowercase all words                               |
