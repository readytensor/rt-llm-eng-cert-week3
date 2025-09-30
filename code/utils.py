import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def count_trainable_params(model: torch.nn.Module) -> int:
    """
    Return the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_gb(model: torch.nn.Module) -> float:
    """
    Calculate the model size in GB based on parameter count and data type.

    Args:
        model: PyTorch model

    Returns:
        Model size in GB
    """
    total_size = 0

    for param in model.parameters():
        # Get the number of elements
        param_size = param.numel()

        # Get the size of each element in bytes based on data type
        if param.dtype == torch.float32:
            bytes_per_param = 4
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bytes_per_param = 2
        elif param.dtype == torch.int8:
            bytes_per_param = 1
        elif param.dtype == torch.float64:
            bytes_per_param = 8
        else:
            # Default to 4 bytes for unknown types
            bytes_per_param = 4

        total_size += param_size * bytes_per_param

    # Convert bytes to GB (1 GB = 1024^3 bytes)
    size_gb = total_size / (1024**3)
    return size_gb


def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return the contents as a dictionary.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def get_last_checkpoint_path(checkpoints_dir: str) -> str:
    """
    Get the path to the last checkpoint in the checkpoints directory.
    """
    checkpoints = sorted(
        [
            int(f.replace("checkpoint-", ""))
            for f in os.listdir(checkpoints_dir)
            if f.startswith("checkpoint")
        ]
    )
    return os.path.join(checkpoints_dir, f"checkpoint-{checkpoints[-1]}")


def push_to_hub(
    model: torch.nn.Module, tokenizer: AutoTokenizer, model_name: str, hf_username: str
):
    """
    Push a model and tokenizer to Hugging Face Hub.
    """
    model_id = f"{hf_username}/{model_name}"
    try:
        model.push_to_hub(model_id, private=False)
        tokenizer.push_to_hub(model_id)
        print(f"Adapters successfully pushed to: https://huggingface.co/{model_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        print("Make sure you're logged in with: huggingface-cli login")


def load_fine_tuned_model(base_model_name: str, adapter_model_name: str):
    """
    Load a fine-tuned model with LoRA adapters from Hugging Face Hub

    Args:
        base_model_name: Original model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        adapter_model_name: Your adapter model name (e.g., "username/model-name")
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading tokenizer from: {adapter_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

    print(f"Loading LoRA adapters from: {adapter_model_name}")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)

    print("✅ Fine-tuned model loaded successfully!")
    return model, tokenizer
