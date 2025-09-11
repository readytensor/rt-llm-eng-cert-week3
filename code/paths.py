import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")
DATASET_FILE = os.path.join(ROOT_DIR, "data", "dataset.jsonl")
