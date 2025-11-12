"""
run_grid_search.py
Simple grid search for LoRA hyperparameters using the SAMSum fine-tuning setup.
"""

import os
import json
from paths import CONFIG_FILE_PATH, OUTPUTS_DIR
from copy import deepcopy
from train_qlora import main as train_main
from utils.config_utils import load_config

# ---------------------------------------------------------------------------
# Load base config
# ---------------------------------------------------------------------------
BASE_CFG = load_config(CONFIG_FILE_PATH)


# ---------------------------------------------------------------------------
# Define default values and search space for one-at-a-time variation
# ---------------------------------------------------------------------------
DEFAULTS = {
    "lora_r": 8,
    "learning_rate": 2e-4,
    "target_modules": ["q_proj", "v_proj"],
}

SEARCH_SPACE = {
    "lora_r": [4, 8, 16, 32],
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


# ---------------------------------------------------------------------------
# Run one-at-a-time parameter search
# ---------------------------------------------------------------------------
def run_grid_search():
    """
    Run one-at-a-time parameter variation experiments.
    Automatically generates experiments by varying one parameter at a time.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    configs = []

    def create_config(param_values, exp_name):
        """Create a config with specified parameter values."""
        cfg = deepcopy(BASE_CFG)

        # Apply parameter values to config
        cfg["lora_r"] = param_values["lora_r"]
        cfg["lora_alpha"] = param_values["lora_r"] * 2
        cfg["learning_rate"] = param_values["learning_rate"]
        cfg["target_modules"] = param_values["target_modules"]
        cfg["wandb_run_name"] = exp_name

        return cfg

    def get_modules_tag(targets):
        """Generate a short tag for target modules."""
        if targets == ["q_proj", "v_proj"]:
            return "qv"
        elif len(targets) == 4:
            return "attn"
        else:
            return "attn_mlp"

    def format_exp_name(param_values, varied_param=None):
        """Generate experiment name from parameter values."""
        parts = []
        if varied_param:
            parts.append(f"vary_{varied_param}")
        else:
            parts.append("baseline")

        parts.append(f"r{param_values['lora_r']}")
        parts.append(f"lr{param_values['learning_rate']}")
        parts.append(get_modules_tag(param_values["target_modules"]))

        return "_".join(parts)

    # 1. Baseline experiment (all defaults)
    baseline_name = format_exp_name(DEFAULTS)
    baseline_cfg = create_config(DEFAULTS, baseline_name)
    configs.append(baseline_cfg)

    # 2. Vary each parameter one at a time
    for param_name in SEARCH_SPACE.keys():
        for value in SEARCH_SPACE[param_name]:
            # Skip if this is the default value
            if value == DEFAULTS[param_name]:
                continue

            # Create param values with all defaults except the one we're varying
            param_values = deepcopy(DEFAULTS)
            param_values[param_name] = value

            exp_name = format_exp_name(param_values, varied_param=param_name)
            cfg = create_config(param_values, exp_name)
            configs.append(cfg)

    print(f"\nRunning {len(configs)} experiments...\n")

    # Run all experiments sequentially
    for i, cfg in enumerate(configs, 1):
        print("=" * 80)
        print(f"Experiment {i}/{len(configs)}: {cfg['wandb_run_name']}")
        print("=" * 80)

        exp_dir = os.path.join(OUTPUTS_DIR, "grid_search", cfg["wandb_run_name"])
        os.makedirs(exp_dir, exist_ok=True)
        cfg["save_dir"] = exp_dir

        model_file_path = os.path.join(
            exp_dir, "lora_adapters", "adapter_model.safetensors"
        )
        if os.path.exists(model_file_path):
            print(f"✗ Model file already exists: {model_file_path}. Skipping...")
            continue

        # Save config for this experiment
        cfg_path = os.path.join(exp_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # Point train_lora.py to this new config
        os.environ["CONFIG_PATH"] = cfg_path

        try:
            train_main(cfg_path=cfg_path)
        except Exception as e:
            print(f"✗ Error in experiment {cfg['wandb_run_name']}: {e}")
            continue

        print(f"✓ Completed: {cfg['wandb_run_name']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_grid_search()
