import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.join(ROOT_DIR, "code")

CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

DATASET_FILE = os.path.join(ROOT_DIR, "dataset.jsonl")

CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

DEEP_SPEED_CONFIG_DIR = os.path.join(CODE_DIR, "lesson3", "deepspeed_config")

DEEP_SPEED_ZERO1_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero1.json")

DEEP_SPEED_ZERO2_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero2.json")

DEEP_SPEED_ZERO3_CONFIG = os.path.join(DEEP_SPEED_CONFIG_DIR, "zero3.json")
