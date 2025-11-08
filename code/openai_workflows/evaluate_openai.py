"""
evaluate_openai.py
Benchmark OpenAI models (base or fine-tuned) on summarization datasets (e.g., SAMSum).
Computes ROUGE metrics and saves predictions + results.
"""

import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils.config_utils import load_config
from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from utils.inference_utils import compute_rouge
from paths import OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
client = OpenAI()

# ---------------------------------------------------------------------------
# OpenAI Inference
# ---------------------------------------------------------------------------


def generate_openai_predictions(
    model_name,
    dataset,
    task_instruction,
    num_samples=None,
    max_workers=10,
    sleep_time=0.25,
):
    """
    Generate predictions from an OpenAI model using concurrent threads.

    Args:
        model_name (str): OpenAI model name or fine-tuned model ID.
        dataset: HF dataset split.
        task_instruction (str): Task-level instruction.
        num_samples (int | None): Limit number of samples.
        max_workers (int): Number of parallel threads (default 5).
        sleep_time (float): Delay between batch submissions (to respect rate limits).

    Returns:
        list[str]: Generated summaries.
    """
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    total = len(dataset)
    preds = [None] * total  # Preallocate list

    def process_sample(idx, sample):
        """Single-sample inference."""
        messages = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=128,
            )
            return idx, response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Error on sample {idx}: {e}")
            return idx, ""

    print(
        f"🚀 Launching parallel inference with {max_workers} workers "
        f"for {total} samples at {sleep_time} seconds between completions..."
    )
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, i, dataset[i]) for i in range(total)]

        for i, future in enumerate(as_completed(futures), start=1):
            idx, output_text = future.result()
            preds[idx] = output_text
            if i % 10 == 0 or i == total:
                print(f"✅ Completed {i}/{total} samples")

            time.sleep(sleep_time)  # Slight delay between completions (safety buffer)

    print(f"⏱️  Inference finished in {time.time() - start_time:.2f} seconds")
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_openai_model(
    model_name: str, cfg, limit: int = None, max_workers: int = 5
):
    """
    Run evaluation on an OpenAI model (base or fine-tuned).

    Args:
        model_name (str): Model name or fine-tuned model ID.
        cfg (dict): Loaded YAML configuration.
        limit (int): Optional sample limit.
        max_workers (int): Number of parallel threads.
    """
    print(f"\n🚀 Evaluating OpenAI model: {model_name}")

    # Dataset configuration
    dataset_cfg = {
        "name": cfg["datasets"][0]["path"],
        "train_samples": cfg.get("train_samples", "all"),
        "val_samples": cfg.get("val_samples", 200),
        "test_samples": cfg.get("test_samples", 200),
        "seed": cfg.get("seed", 42),
        "cache_dir": cfg["datasets"][0].get("cache_dir", None),
    }

    _, val_data, _ = load_and_prepare_dataset(dataset_cfg)
    if limit:
        val_data = val_data.select(range(limit))
        print(f"⚙️  Limiting evaluation to first {limit} samples.")

    # Generate predictions (parallelized)
    preds = generate_openai_predictions(
        model_name=model_name,
        dataset=val_data,
        task_instruction=cfg["task_instruction"],
        num_samples=len(val_data),
        max_workers=max_workers,
        sleep_time=0.1,
    )

    # Compute ROUGE
    print("\n📈 Computing ROUGE scores...")
    scores = compute_rouge(preds, val_data)
    print("\nROUGE Summary:")
    print(f"  ROUGE-1: {scores['rouge1']:.2%}")
    print(f"  ROUGE-2: {scores['rouge2']:.2%}")
    print(f"  ROUGE-L: {scores['rougeL']:.2%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    model_safe = model_name.replace("/", "_").replace(":", "_")
    output_dir = os.path.join(OUTPUTS_DIR, "openai", model_safe)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "model_name": model_name,
        "num_samples": len(val_data),
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
    }

    results_path = os.path.join(output_dir, "eval_results.json")
    preds_path = os.path.join(output_dir, "predictions.jsonl")

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

    print(f"\n✅ Evaluation complete for {model_name}")
    print(f"💾 Results saved to: {results_path}")
    print(f"💾 Predictions saved to: {preds_path}")

    return scores, preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI model on summarization dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name or fine-tuned model ID",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of validation samples"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of parallel inference threads",
    )
    args = parser.parse_args()

    print("⚙️  Loading configuration...")
    cfg = load_config()

    rouge_scores, predictions = evaluate_openai_model(
        model_name=args.model,
        cfg=cfg,
        limit=args.limit,
        max_workers=args.max_workers,
    )

    print("\nExample prediction:\n")
    print(predictions[0])
    print("\nRouge scores:\n")
    print(rouge_scores)

    print("Completed successfully.")
