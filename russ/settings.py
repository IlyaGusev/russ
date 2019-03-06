import os
from pkg_resources import resource_filename

DATA_DIR = resource_filename(__name__, "data")
RU_WIKTIONARY_DICT = os.path.join(DATA_DIR, "ru_wiki.txt")
RU_INFLECTED_DICT = os.path.join(DATA_DIR, "ru_inflected.txt")
RU_CUSTOM_DICT = os.path.join(DATA_DIR, "ru_custom.txt")
RU_ALL_DICT = os.path.join(DATA_DIR, "ru_all.txt")
