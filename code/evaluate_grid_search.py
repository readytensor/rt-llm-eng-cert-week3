import os
import json
from code.paths import GRIDSEARCH_OUTPUTS_DIR
from code.evaluate_qlora import evaluate_peft_model


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

            evaluate_peft_model(cfg, adapter_dir, results_dir)
