"""
prepare_openai_jsonl.py
Converts a Hugging Face dataset (e.g., SAMSum) into OpenAI fine-tuning JSONL format.
"""

import os
import json
from datasets import load_dataset
from paths import DATASETS_DIR
from utils.config_utils import load_config


def prepare_openai_jsonl(dataset_name: str, task_instruction: str, output_dir: str):
    """
    Converts the Hugging Face dataset into OpenAI-compatible fine-tuning files.
    Creates train.jsonl and validation.jsonl inside the given output directory.

    Args:
        dataset_name (str): Dataset ID (e.g., "knkarthick/samsum")
        task_instruction (str): Instruction to prepend in the "system" message.
        output_dir (str): Directory to save JSONL files.
    """
    print(f"📥 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    def to_openai_format(example):
        """Convert one sample into OpenAI chat format."""
        return {
            "messages": [
                {"role": "system", "content": task_instruction},
                {"role": "user", "content": example["dialogue"]},
                {"role": "assistant", "content": example["summary"]},
            ]
        }

    for split in ["train", "validation"]:
        if split not in dataset:
            print(f"⚠️  No '{split}' split found, skipping...")
            continue

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in dataset[split]:
                record = to_openai_format(ex)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ Saved {split} split to {out_path} ({len(dataset[split])} samples)")


if __name__ == "__main__":
    cfg = load_config()
    dataset_name = cfg["datasets"][0]["path"]
    task_instruction = cfg["task_instruction"]

    safe_name = dataset_name.replace("/", "_")
    output_dir = os.path.join(DATASETS_DIR, safe_name)

    prepare_openai_jsonl(dataset_name, task_instruction, output_dir)
