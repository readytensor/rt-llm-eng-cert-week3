"""
evaluate_baseline.py
Evaluate the base (unfine-tuned) model on the SAMSum dataset to establish baseline ROUGE scores.
"""

import os
import yaml
import json
import torch
from dotenv import load_dotenv

# --- Internal Imports ---
from paths import CONFIG_FILE_PATH, BASELINE_OUTPUTS_DIR
from utils.data_utils import load_and_prepare_dataset
from utils.inference_utils import generate_predictions, compute_rouge
from utils.model_utils import setup_model_and_tokenizer
from utils.config_utils import load_config


load_dotenv()
os.makedirs(BASELINE_OUTPUTS_DIR, exist_ok=True)

cfg = load_config()

def evaluate_baseline():
    """Run baseline evaluation on the SAMSum dataset using the base model."""

    # Load validation data
    _, val_data, _ = load_and_prepare_dataset(cfg)
    print(f"📊 Loaded {len(val_data)} validation samples.")

    # Load model + tokenizer (no quantization or LoRA)
    model, tokenizer = setup_model_and_tokenizer(
        cfg=cfg,
        use_4bit=False,
        use_lora=False,
    )

    # Generate predictions
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        batch_size=4,
    )

    # Compute ROUGE metrics
    scores = compute_rouge(preds, val_data)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    results = {
        "model_name": cfg["base_model"],
        "num_samples": len(val_data),
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }

    results_path = os.path.join(BASELINE_OUTPUTS_DIR, "eval_results.json")
    preds_path = os.path.join(BASELINE_OUTPUTS_DIR, "predictions.jsonl")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
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

    print(f"\n💾 Saved results to: {results_path}")
    print(f"💾 Saved predictions to: {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting baseline evaluation...")
    rouge_scores, predictions = evaluate_baseline()
    print("\n✅ Evaluation complete.")

    
    print("\n📈 Baseline ROUGE Results:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.2%}")

    print("\nExample prediction:\n")
    print(predictions[0])
    print("\nRouge scores:\n")
    print(rouge_scores)
