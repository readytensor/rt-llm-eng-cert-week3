"""
openai_workflow.py
Interactive controller script for the OpenAI fine-tuning workflow.
Now directly imports and calls internal functions (no subprocess overhead).
"""

import os
from utils.config_utils import load_config
from paths import OPENAI_FILES_DIR
from openai_workflows.prepare_openai_jsonl import prepare_openai_jsonl
from openai_workflows.openai_finetune_runner import main as run_finetune_main
from openai_workflows.evaluate_openai import evaluate_openai_model


# ---------------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------------


def run_prepare_data():
    """Prepare JSONL files for OpenAI fine-tuning."""
    print("\n📦 Preparing OpenAI fine-tuning data...")
    cfg = load_config()
    dataset_name = cfg["datasets"][0]["path"]
    task_instruction = cfg["task_instruction"]

    safe_name = dataset_name.replace("/", "_")
    output_dir = os.path.join(OPENAI_FILES_DIR, safe_name)

    prepare_openai_jsonl(dataset_name, task_instruction, output_dir)
    print(f"\n✅ Data prepared and saved to: {output_dir}")


def run_finetune():
    """Run fine-tuning job."""
    print("\n⚙️  Running OpenAI fine-tuning job...")
    # Just call the runner’s main() so it handles defaults and CLI logic
    run_finetune_main()
    print("\n✅ Fine-tuning workflow complete!")


def run_evaluation():
    """Run model evaluation (base or fine-tuned)."""
    print("\n🧠 Running OpenAI model evaluation...")
    cfg = load_config()
    model_name = input(
        "Enter OpenAI model name (e.g., 'gpt-4o-mini' or fine-tuned ID): "
    ).strip()

    if not model_name:
        print("❌ Model name cannot be empty.")
        return

    evaluate_openai_model(model_name, cfg)
    print(f"\n✅ Evaluation complete for model: {model_name}")


# ---------------------------------------------------------------------------
# Main Menu Loop
# ---------------------------------------------------------------------------


def main():
    print("\n🚀 Starting OpenAI workflow...")
    while True:
        print("\n========================================")
        print("  🧩 OpenAI Fine-Tuning Workflow")
        print("========================================")
        print("1️⃣  Prepare dataset for fine-tuning")
        print("2️⃣  Run fine-tuning job")
        print("3️⃣  Evaluate base or fine-tuned model")
        print("4️⃣  Exit")
        print("========================================")

        choice = input("Select an option (1/2/3/4): ").strip()

        if choice == "1":
            run_prepare_data()
        elif choice == "2":
            run_finetune()
        elif choice == "3":
            run_evaluation()
        elif choice == "4":
            print("\n👋 Exiting OpenAI workflow.")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
