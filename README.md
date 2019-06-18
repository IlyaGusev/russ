# Python library for word stress detection #

[![Build Status](https://travis-ci.org/IlyaGusev/russ.svg?branch=master)](https://travis-ci.org/IlyaGusev/russ)
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



### Commands

#### download.sh

Script for downloading datasets:
ru_custom.txt: 885 words
zaliznyak.txt: 86839 lexemes
ruwiktionary-20190501-pages-articles.xml: articles from ruwiktionary

#### prepare_data.py

Script preparing data for training

| Argument               | Default | Description                                                                         |
|:-----------------------|:--------|:------------------------------------------------------------------------------------|
| --wiktionary-dump-path | None    | path to downloaded wiktionary dump                                                  |
| --custom-dict-path     | None    | path to file with custom words                                                      |
| --inflected-dict-path  | None    | path to downloaded file with lexemes                                                |
| --split-lexemes        | False   | use only file with lexemes and leave different lexemes in train, val and test files |
| --train-path           |         | path to output train file                                                           |
| --val-path             |         | path to output validation file                                                      |
| --test-path            |         | path to output test file                                                            |
| --val-part             | 0.05    | part of validation file                                                             |
| --test-part            | 0.05    | part of test file                                                                   |
| --lower                | Fasle   | lowercase all words. Order: lower -> sort -> shuffle                                |
| --sort                 | False   | sort all words or lexemes                                                           |
| --shuffle              | False   | shuffle all words or lexemes                                                        |


#### train.py

Script for model training. Model directory should exist as well as config file and vocabulary directory.

| Argument          | Default | Description                          |
|:------------------|:--------|:-------------------------------------|
| --train-path      |         | path to train dataset                |
| --model-path      |         | path to directory with model's files |
| --val-path        | None    | path to val dataset                  |
| --seed            | 1048596 | random seed                          |
| --vocabulary-path | None    | custom path to vocabulary            |
| --config-path     | None    | custom path to config                |

#### evaluate.py

Script for model evaluation.

| Argument             | Default | Description                                               |
|:---------------------|:--------|:----------------------------------------------------------|
| --test-path          |         | path to test dataset                                      |
| --model-path         |         | path to directory with model's files                      |
| --metric             | wer     | what metric to evaluate, choices=("wer", "wer-constr")    |
| --max-count          | None    | how many test examples to consider                        |
| --report-every       | None    | print metrics every N'th step                             |
| --batch-size         | 32      | size of a batch with test examples to run simultaneously  |
| --errors-file-path   | None    | path to log file with all errors                          |
