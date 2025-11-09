"""
run_grid_search.py
Simple grid search for LoRA hyperparameters using the SAMSum fine-tuning setup.
"""

import os
import json
from itertools import product
from copy import deepcopy
from train_lora import main as train_main


# ---------------------------------------------------------------------------
# Load base config
# ---------------------------------------------------------------------------
with open("configs/config_samsum.json", encoding="utf-8") as f:
    BASE_CFG = json.load(f)


# ---------------------------------------------------------------------------
# Define search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "lora_r": [4, 8, 16],
    "lora_alpha": [8, 16, 32],
    "learning_rate": [2e-4, 2e-5],
    "target_modules": [
        ["q_proj", "v_proj"],  # Just Q and V
        ["q_proj", "v_proj", "k_proj", "o_proj"],  # All attention
        [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Attn + MLP
    ],
}

OUTPUT_ROOT = "./experiments"


# ---------------------------------------------------------------------------
# Run grid search
# ---------------------------------------------------------------------------
def run_grid_search():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    configs = []

    # Create all combinations of parameters
    for r, alpha, lr, targets in product(
        SEARCH_SPACE["lora_r"],
        SEARCH_SPACE["lora_alpha"],
        SEARCH_SPACE["learning_rate"],
        SEARCH_SPACE["target_modules"],
    ):
        cfg = deepcopy(BASE_CFG)
        cfg["lora"]["r"] = r
        cfg["lora"]["alpha"] = alpha
        cfg["training"]["learning_rate"] = lr
        cfg["lora"]["target_modules"] = targets

        # Create readable experiment name
        modules_tag = (
            "qv"
            if targets == ["q_proj", "v_proj"]
            else "attn" if len(targets) == 4 else "attn_mlp"
        )
        cfg["experiment_name"] = f"r{r}_a{alpha}_lr{lr}_{modules_tag}"
        configs.append(cfg)

    print(f"\nRunning {len(configs)} experiments...\n")

    # Run all experiments sequentially
    for i, cfg in enumerate(configs, 1):
        print("=" * 80)
        print(f"Experiment {i}/{len(configs)}: {cfg['experiment_name']}")
        print("=" * 80)

        exp_dir = os.path.join(OUTPUT_ROOT, cfg["experiment_name"])
        os.makedirs(exp_dir, exist_ok=True)

        # Save config for this experiment
        cfg_path = os.path.join(exp_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # Point train_lora.py to this new config
        os.environ["CONFIG_PATH"] = cfg_path

        try:
            train_main()
        except Exception as e:
            print(f"✗ Error in experiment {cfg['experiment_name']}: {e}")
            continue

        print(f"✓ Completed: {cfg['experiment_name']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_grid_search()
