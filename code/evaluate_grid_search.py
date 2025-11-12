import os
import json
from paths import GRIDSEARCH_OUTPUTS_DIR
from evaluate_qlora import evaluate_peft_model


def evaluate_grid_search():
    for exp_dir in os.listdir(GRIDSEARCH_OUTPUTS_DIR):
        if os.path.isdir(os.path.join(GRIDSEARCH_OUTPUTS_DIR, exp_dir)):
            print(f"Evaluating {exp_dir}")
            adapter_dir = os.path.join(GRIDSEARCH_OUTPUTS_DIR, exp_dir, "lora_adapters")
            with open(
                os.path.join(GRIDSEARCH_OUTPUTS_DIR, exp_dir, "config.json"), "r"
            ) as f:
                cfg = json.load(f)
            results_dir = os.path.join(GRIDSEARCH_OUTPUTS_DIR, exp_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, "eval_results.json")
            if os.path.exists(results_file):
                print(f"Results already exist for {exp_dir}. Skipping...")
                continue

            evaluate_peft_model(cfg, adapter_dir, results_dir)


if __name__ == "__main__":
    evaluate_grid_search()
