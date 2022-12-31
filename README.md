# Python library for word stress detection #

[![Tests Status](https://github.com/IlyaGusev/russ/actions/workflows/python-package.yml/badge.svg)](https://github.com/IlyaGusev/russ/actions/workflows/python-package.yml)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/russ/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/russ)
[![codecov](https://codecov.io/gh/IlyaGusev/russ/branch/master/graph/badge.svg)](https://codecov.io/gh/IlyaGusev/russ)

### Install
```
pip install russ
```

or

```
git clone https://github.com/IlyaGusev/russ
cd russ
git lfs pull
pip install -r requirements.txt
python setup.py
```

### Usage

```
from russ.stress.predictor import StressPredictor

model = StressPredictor()
model.predict("корова")

>>> [3]
```

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
| --train-path           |         | path to output train file                                                           |
| --val-path             |         | path to output validation file                                                      |
| --test-path            |         | path to output test file                                                            |
| --val-part             | 0.05    | part of validation file                                                             |
| --test-part            | 0.05    | part of test file                                                                   |
| --lower                | Fasle   | lowercase all words. Order: lower -> sort -> shuffle                                |
