import os
import json
import torch
from datasets import Dataset
import pandas as pd
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.data_utils import load_and_prepare_dataset, build_messages_for_sample
from train_lora import preprocess_samples
from utils.model_utils import setup_model_and_tokenizer
from utils.config_utils import load_config

# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CFG = load_config()

MODEL_NAME = CFG["base_model"]
TASK_INSTRUCTION = CFG["task_instruction"]
CFG_LORA = CFG["lora"] if "lora" in CFG else {}
CFG_DATASET = CFG["dataset"]
CFG_QUANT = CFG["quantization"] if "quantization" in CFG else {}

OUTPUT_DIR = CFG.get("output_dir", "./outputs/lora_samsum")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------------------------

model, tokenizer = setup_model_and_tokenizer(cfg=CFG, use_4bit=True, use_lora=False)

# ---------------------------------------------------------------------------
# Create a simple one-sample dataset for debugging
# ---------------------------------------------------------------------------

example = {
    "dialogue": "A: Hi!\nB: Hello! How are you?\nA: I'm great, thanks!",
    "summary": "A greets B and says they're doing well.",
}

debug_data = Dataset.from_dict({
    "dialogue": [example["dialogue"]],
    "summary": [example["summary"]],
})

# ---------------------------------------------------------------------------
# Tokenize the example using preprocess_samples()
# ---------------------------------------------------------------------------

tokenized_debug = preprocess_samples(
    debug_data,
    tokenizer=tokenizer,
    task_instruction=TASK_INSTRUCTION,
    max_length=CFG.get("sequence_len", 512),
)

# ---------------------------------------------------------------------------
# Reconstruct and print original input prompt
# ---------------------------------------------------------------------------

# Use the same logic as your message-building function
prompt_text = (
    f"{TASK_INSTRUCTION.strip()}\n\n"
    f"## Dialogue:\n{example['dialogue']}\n\n"
    "## Summary:"
)

print("\n============================")
print("📜 ORIGINAL PROMPT SENT TO TOKENIZER")
print("============================")
print(prompt_text)
print("============================\n")
print(f"Ground truth summary: \n{example['summary']}\n")
print("============================\n")

# ---------------------------------------------------------------------------
# Visualize tokenization and masking
# ---------------------------------------------------------------------------

# Extract first example
ex = {k: v[0] for k, v in tokenized_debug.items()}
tokens = [tokenizer.decode([tid]) for tid in ex["input_ids"]]

df = pd.DataFrame({
    "token": tokens,
    "input_id": ex["input_ids"],
    "attention_mask": ex["attention_mask"],
    "label": ex["labels"],
})
df["masked"] = df["label"].apply(lambda x: x == -100)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

print(df.head(100))
