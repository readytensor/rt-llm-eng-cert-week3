"""
evaluate_lora.py
Run inference and evaluation on the SAMSum dataset using a fine-tuned LoRA model.
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from dotenv import load_dotenv

from utils import load_and_prepare_dataset, build_messages_for_sample, compute_rouge


load_dotenv()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("config.json") as f:
    CFG = json.load(f)

MODEL_NAME = CFG["model_name"]
TASK_INSTRUCTION = CFG["task_instruction"]
CFG_DATASET = CFG["dataset"]

ADAPTER_DIR = "./outputs/lora_samsum/lora_adapters"
OUTPUT_DIR = "./outputs/lora_samsum"


# ---------------------------------------------------------------------------
# Generate summaries
# ---------------------------------------------------------------------------
def generate_predictions(
    model, tokenizer, dataset, task_instruction, num_samples=None, batch_size=8
):
    """Generate summaries for test samples."""
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    prompts = []
    for sample in dataset:
        messages = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=torch.bfloat16,
        do_sample=False,
    )

    preds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
        batch = prompts[i : i + batch_size]
        outputs = pipe(batch, max_new_tokens=128, return_full_text=False)
        preds.extend([o[0]["generated_text"].strip() for o in outputs])

    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    adapter_dir=ADAPTER_DIR, model_name=MODEL_NAME, task_instruction=TASK_INSTRUCTION
):
    """Load LoRA adapters and evaluate on test data."""
    print(f"\nLoading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Loading LoRA adapters from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"

    _, val_data, _ = load_and_prepare_dataset(CFG_DATASET)

    print(f"\nRunning inference on {len(val_data)} samples...")
    preds = generate_predictions(
        model, tokenizer, val_data, task_instruction, batch_size=4
    )

    print("\nComputing ROUGE scores...")
    scores = compute_rouge(preds, val_data)

    print("\nROUGE Results:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    results = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "num_samples": len(val_data),
        "adapter_dir": adapter_dir,
        "model_name": model_name,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    preds_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w") as f:
        for i, pred in enumerate(preds):
            json.dump(
                {
                    "dialogue": val_data[i]["dialogue"],
                    "reference": val_data[i]["summary"],
                    "prediction": pred,
                },
                f,
            )
            f.write("\n")

    print(f"\nSaved results to {results_path}")
    print(f"Saved predictions to {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    scores, preds = evaluate_model()
    print("\nEvaluation complete.")
    print("Sample prediction:\n")
    print(preds[0])


if __name__ == "__main__":
    main()
