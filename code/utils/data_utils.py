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
    Checks local path first, then downloads from Hugging Face if missing.

    Args:
        cfg_dataset (dict): Contains dataset name, sample sizes, seed, and optional cache_dir.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    cfg_dataset = cfg.get("dataset", {})
    if "name" in cfg_dataset:
        dataset_name = cfg_dataset["name"]
    elif "datasets" in cfg and isinstance(cfg["datasets"], list):
        dataset_name = cfg["datasets"][0]["path"]
    else:
        raise KeyError("Dataset name/path not found in config.")
    seed = cfg_dataset.get("seed", 42)
    cache_dir = cfg_dataset.get("cache_dir")
    local_path = get_local_dataset_path(dataset_name, cache_dir)

    # Ensure the datasets directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Try to load from local path first
    if os.path.exists(local_path):
        print(f"📂 Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"⬇️  Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)
        print(f"✅ Dataset saved locally to: {local_path}")

    # Handle split key variations
    val_key = "validation" if "validation" in dataset else "val"

    train = select_subset(dataset["train"], cfg_dataset.get("train_samples", "all"))
    val = select_subset(dataset[val_key], cfg_dataset.get("val_samples", 200))
    test = select_subset(
        dataset["test"].shuffle(seed=seed), cfg_dataset.get("test_samples", 200)
    )

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
