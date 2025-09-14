import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.join(ROOT_DIR, "code")

CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

DATASET_FILE = os.path.join(ROOT_DIR, "dataset.jsonl")

CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
