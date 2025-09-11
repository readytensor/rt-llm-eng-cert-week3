import os
import sys
import json
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import DATASET_FILE


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """
    Load a JSONL dataset from file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries containing the dataset
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def format_instruction_data(data_point: Dict) -> str:
    """
    Format a data point into instruction format for fine-tuning.

    Args:
        data_point: Dictionary with 'question' and 'output' keys

    Returns:
        Formatted instruction string
    """
    question = data_point["question"]
    input = data_point["input"]
    output = data_point["output"]

    formatted_text = f"### Question\n{question}\n\n"

    if input:
        formatted_text += f"### Input\n{input}\n\n"

    formatted_text += f"### Output\n{output}"

    return formatted_text


def prepare_dataset(file_path: str) -> Dataset:
    """
    Load and prepare the dataset for fine-tuning.

    Args:
        file_path: Path to the JSONL dataset file

    Returns:
        HuggingFace Dataset object ready for training
    """
    # Load raw data
    raw_data = load_jsonl_dataset(file_path)

    # Format each data point
    formatted_data = []
    for data_point in tqdm(raw_data, desc="Preparing dataset"):
        formatted_text = format_instruction_data(data_point)
        formatted_data.append({"text": formatted_text})

    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)
    return dataset


def apply_assistant_masking(
    input_ids: List[int], tokenizer: AutoTokenizer
) -> List[int]:
    """
    Apply assistant-only masking by setting instruction tokens to -100.

    Args:
        input_ids: Tokenized input sequence
        tokenizer: The tokenizer used

    Returns:
        Labels with instruction tokens masked (-100)
    """
    labels = input_ids.copy()

    # Find the "### Output" marker to identify where assistant response starts
    output_marker = "### Output"
    output_marker_tokens = tokenizer.encode(output_marker, add_special_tokens=False)

    # Find where the output section begins
    output_start_idx = None
    for i in range(len(input_ids) - len(output_marker_tokens) + 1):
        if input_ids[i : i + len(output_marker_tokens)] == output_marker_tokens:
            output_start_idx = i + len(output_marker_tokens)
            break

    # If we found the output marker, mask everything before it
    if output_start_idx is not None:
        # Mask instruction tokens (set to -100)
        for i in range(output_start_idx):
            labels[i] = -100

    return labels


def tokenize_dataset(
    model_name: str,
    dataset_file: str = DATASET_FILE,
    assistant_only_masking: bool = True,
) -> Tuple[Dataset, AutoTokenizer]:
    """
    Load and prepare the dataset with tokenization and assistant-only masking.

    Args:
        model_name: Name of the model to use for tokenizer
        dataset_file: Path to the dataset file
        assistant_only_masking: Whether to apply assistant-only masking


    Returns:
        tuple: (tokenized_dataset, tokenizer)
    """

    dataset = prepare_dataset(dataset_file)
    print(f"Dataset loaded: {len(dataset)} examples")

    print("\nFirst example:")
    print(dataset[0]["text"])

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset with assistant masking
    def tokenize_and_mask_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None,
        )

        # Apply assistant-only masking to each example
        labels = []
        for input_ids in tokenized["input_ids"]:
            if assistant_only_masking:
                masked_labels = apply_assistant_masking(input_ids, tokenizer)
            else:
                masked_labels = input_ids
            labels.append(masked_labels)

        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_and_mask_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset, tokenizer
