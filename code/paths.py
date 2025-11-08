import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.join(ROOT_DIR, "code")

CONFIG_FILE_PATH = os.path.join(CODE_DIR, "config.yaml")

DATA_DIR = os.path.join(ROOT_DIR, "data")

DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

BASELINE_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "baseline")
LORA_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "lora")
GRIDSEARCH_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "grid_search")


MODEL_DIR = os.path.join(DATA_DIR, "model")

EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")

OPENAI_FILES_DIR = os.path.join(EXPERIMENTS_DIR, "openai_files")

WANDB_DIR = os.path.join(DATA_DIR, "wandb")

# DATASET_FILE = os.path.join(ROOT_DIR, "dataset.jsonl")

# CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

# DEEP_SPEED_CONFIG_DIR = os.path.join(CODE_DIR, "lesson3", "deepspeed_config")

# DEEP_SPEED_ZERO1_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero1.json")

# DEEP_SPEED_ZERO2_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero2.json")

# DEEP_SPEED_ZERO3_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero3.json")
