import os
from pkg_resources import resource_filename

MODELS_DIR = resource_filename(__name__, "models")
RU_MAIN_MODEL = os.path.join(MODELS_DIR, "ru-main")

CONFIGS_DIR = resource_filename(__name__, "configs")
BASIC_CONFIG = os.path.join(CONFIGS_DIR, "base_config.json")