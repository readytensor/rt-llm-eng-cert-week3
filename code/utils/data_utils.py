"""
data_utils.py
Utility functions for loading datasets and preparing text samples for training or inference.
Uses shared paths from paths.py for dataset caching and supports optional cache_dir from config.
"""

import os
from datasets import load_dataset, load_from_disk
from paths import DATASETS_DIR


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------


def get_local_dataset_path(dataset_name: str, cache_dir: str = None) -> str:
    """
    Build a safe local path for storing datasets based on their Hugging Face name.

    Args:
        dataset_name (str): Hugging Face dataset identifier (e.g., 'knkarthick/samsum').
        cache_dir (str | None): Optional cache directory override (e.g., from config).

    Returns:
        str: Absolute path to local dataset folder.
    """
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    base_dir = cache_dir or DATASETS_DIR
    return os.path.join(base_dir, safe_name)


def select_subset(split, n):
    """Return a subset of n samples, or the entire split if n='all' or None."""
    if n == "all" or n is None:
        return split
    return split.select(range(min(n, len(split))))


def load_and_prepare_dataset(cfg):
    """
    Load dataset splits according to configuration.
    Supports both old-style ("dataset": {...}) and new-style ("datasets": [ {...} ]) configs.
    """
    # Determine dataset name and sampling settings
    if "dataset" in cfg:
        cfg_dataset = cfg["dataset"]
        dataset_name = cfg_dataset["name"]
    elif "datasets" in cfg and isinstance(cfg["datasets"], list):
        cfg_dataset = cfg["datasets"][0]
        dataset_name = cfg_dataset["path"]
    else:
        raise KeyError("Dataset name/path not found in configuration.")

    seed = cfg.get("seed", 42)
    local_path = os.path.join(DATASETS_DIR, dataset_name.replace("/", "_"))

    # Sample sizes (use top-level config if available)
    n_train = cfg.get("train_samples", "all")
    n_val = cfg.get("val_samples", 200)
    n_test = cfg.get("test_samples", 200)

    # Try loading locally, else download
    os.makedirs(DATASETS_DIR, exist_ok=True)
    if os.path.exists(local_path):
        print(f"📂 Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"⬇️  Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)
        print(f"✅ Dataset saved locally to: {local_path}")

    # Handle variations in split keys
    val_key = "validation" if "validation" in dataset else "val"

    train = select_subset(dataset["train"], n_train)
    val = select_subset(dataset[val_key], n_val)
    test = select_subset(dataset["test"].shuffle(seed=seed), n_test)

    print(f"📊 Loaded {len(train)} train / {len(val)} val / {len(test)} test samples.")
    return train, val, test


# ---------------------------------------------------------------------------
# Prompt / Message Construction
# ---------------------------------------------------------------------------


def build_user_prompt(dialogue: str, task_instruction: str) -> str:
    """Construct a summarization-style prompt given a dialogue and instruction."""
    return f"{task_instruction}\n\n## Dialogue:\n{dialogue}\n## Summary:"


def build_messages_for_sample(sample, task_instruction, include_assistant=False):
    """
    Build a chat-style message list for a given sample, compatible with
    models that use chat templates (like Llama 3).
    """
    messages = [
        {
            "role": "user",
            "content": build_user_prompt(sample["dialogue"], task_instruction),
        }
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": sample["summary"]})
    return messages
