#!/usr/bin/env python3
"""
Script to push a local model directory to Hugging Face Hub.

Usage:
    python push_to_hf.py --model-path <path_to_adapters> --hf-model-name <huggingface_repo_name> --base-model-name <base_model>

Example:
    python push_to_hf.py --model-path ./checkpoints/checkpoint-45 --hf-model-name username/my-model --base-model-name meta-llama/Llama-3.2-1B
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def push_model_to_hf(model_path: str, hf_model_name: str, base_model_name: str):
    """
    Load base model from HF, apply LoRA adapters, and push to Hugging Face Hub.

    Args:
        model_path: Path to the local LoRA adapters directory
        hf_model_name: Name of the Hugging Face repository (format: username/model-name)
        base_model_name: Name of the base model on Hugging Face
    """
    # Validate model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model path '{model_path}' does not exist.")
        sys.exit(1)

    if not model_path.is_dir():
        print(f"Error: Model path '{model_path}' is not a directory.")
        sys.exit(1)

    print(f"Loading base model '{base_model_name}' from Hugging Face...")

    try:
        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        print(f"Loading LoRA adapters from '{model_path}'...")

        # Load adapters and create adapted model
        model = PeftModel.from_pretrained(base_model, model_path)

        print(f"Pushing adapted model to Hugging Face Hub as '{hf_model_name}'...")

        # Push to hub
        model.push_to_hub(hf_model_name)
        tokenizer.push_to_hub(hf_model_name)

        print(
            f"\n✓ Successfully pushed model to https://huggingface.co/{hf_model_name}"
        )

    except Exception as e:
        print(f"\nError: Failed to push model to Hugging Face Hub.")
        print(f"Details: {e}")
        print("\nMake sure you are logged in to Hugging Face. Run:")
        print("  huggingface-cli login")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Push a local model with LoRA adapters to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python push_to_hf.py --model-path ./checkpoints/checkpoint-45 --hf-model-name username/my-model --base-model-name meta-llama/Llama-3.2-1B
  python push_to_hf.py --model-path /path/to/adapters --hf-model-name org/model-name --base-model-name google/gemma-2b
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local LoRA adapters directory",
    )

    parser.add_argument(
        "--hf-model-name",
        type=str,
        required=True,
        help="Hugging Face repository name (format: username/model-name)",
    )

    parser.add_argument(
        "--base-model-name",
        type=str,
        required=True,
        help="Base model name on Hugging Face (e.g., meta-llama/Llama-3.2-1B)",
    )

    args = parser.parse_args()

    push_model_to_hf(args.model_path, args.hf_model_name, args.base_model_name)


if __name__ == "__main__":
    main()
