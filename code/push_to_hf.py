#!/usr/bin/env python3
"""
Script to push a local model directory to Hugging Face Hub.

Usage:
    python push_to_hf.py --model-path <path_to_model> --hf-model-name <huggingface_repo_name>

Example:
    python push_to_hf.py --model-path ./experiments/my_model --hf-model-name username/my-model
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM


def push_model_to_hf(model_path: str, hf_model_name: str):
    """
    Load model from local path and push to Hugging Face Hub.

    Args:
        model_path: Path to the local model directory
        hf_model_name: Name of the Hugging Face repository (format: username/model-name)
    """
    # Validate model path exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model path '{model_path}' does not exist.")
        sys.exit(1)

    if not model_path.is_dir():
        print(f"Error: Model path '{model_path}' is not a directory.")
        sys.exit(1)

    print(f"Loading model from '{model_path}'...")

    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path)

        print(f"Pushing to Hugging Face Hub as '{hf_model_name}'...")

        # Push to hub
        model.push_to_hub(hf_model_name)

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
        description="Push a local model directory to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python push_to_hf.py --model-path ./experiments/my_model --hf-model-name username/my-model
  python push_to_hf.py --model-path /path/to/model --hf-model-name org/model-name
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local model directory to upload",
    )

    parser.add_argument(
        "--hf-model-name",
        type=str,
        required=True,
        help="Hugging Face repository name (format: username/model-name)",
    )

    args = parser.parse_args()

    push_model_to_hf(args.model_path, args.hf_model_name)


if __name__ == "__main__":
    main()
