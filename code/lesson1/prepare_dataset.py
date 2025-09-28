import os
import sys
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_dataset(
    dataset_name: str,
    instruction_column: str,
    input_column: str,
    output_column: str,
    sample_size: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare the dataset for fine-tuning.

    Args:
        file_path: Path to the JSONL dataset file

    Returns:
        HuggingFace Dataset object ready for training
    """

    def format_instruction_data(data_point: Dict) -> str:
        """
        Format a data point into instruction format for fine-tuning.

        Args:
            data_point: Dictionary with 'instruction', 'input', and 'output' keys

        Returns:
            Formatted instruction string
        """
        instruction = data_point[instruction_column]
        input = data_point[input_column]
        output = data_point[output_column]

        formatted_text = f"### Instruction\n{instruction}\n\n"

        if input:
            formatted_text += f"### Input\n{input}\n\n"

        formatted_text += f"### Output\n{output}"

        return {"text": formatted_text}

    dataset = load_dataset(dataset_name)

    if sample_size is not None:
        dataset["train"] = dataset["train"].select(range(sample_size))

    train_dataset = dataset["train"] if "train" in dataset else None
    validation_dataset = dataset["validation"] if "validation" in dataset else None
    test_dataset = dataset["test"] if "test" in dataset else None

    if train_dataset is not None:
        train_dataset = train_dataset.map(format_instruction_data)

    if validation_dataset is not None:
        validation_dataset = validation_dataset.map(format_instruction_data)

    if test_dataset is not None:
        test_dataset = test_dataset.map(format_instruction_data)

    return train_dataset, validation_dataset, test_dataset


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
    dataset_name: str,
    instruction_column: str,
    input_column: str,
    output_column: str,
    assistant_only_masking: bool = True,
    max_length: int = 2048,
    sample_size: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Load and prepare the dataset with tokenization and assistant-only masking.

    Args:
        model_name: Name of the model to use for tokenizer
        dataset_name: Name of the dataset to use
        assistant_only_masking: Whether to apply assistant-only masking


    Returns:
        tuple: (tokenized_dataset, tokenizer)
    """

    train_dataset, validation_dataset, test_dataset = prepare_dataset(
        dataset_name,
        instruction_column=instruction_column,
        input_column=input_column,
        output_column=output_column,
    )

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset with assistant masking
    def tokenize_and_mask_function(examples):
        texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
        tokenized = tokenizer(
            texts_with_eos,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True,
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

    train = train_dataset.map(
        tokenize_and_mask_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    validation = None
    if validation_dataset is not None:
        validation = validation_dataset.map(
            tokenize_and_mask_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
    test = None
    if test_dataset is not None:
        test = test_dataset.map(
            tokenize_and_mask_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

    return train, validation, test, tokenizer


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Remove labels before tokenizer.pad so only ids/mask are padded
        labels = [f.pop("labels") for f in features]

        # This pads input_ids and attention_mask consistently
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # Now pad labels to the same max length
        max_len = batch["input_ids"].size(1)
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            padded_labels[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        batch["labels"] = padded_labels
        return batch
