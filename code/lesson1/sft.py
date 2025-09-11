import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer
from utils import (
    count_trainable_params,
    read_json_file,
    get_model_size_gb,
)
from prepare_dataset import tokenize_dataset
from paths import CONFIG_FILE

from typing import Optional

config = read_json_file(CONFIG_FILE)
model_name = config["model_name"]

quantization_config = config["quantization_config"]
lora_config = config["lora_config"]
use_qlora = config["use_qlora"]
training_args = config["training_args"]

assistant_only_masking = config["assistant_only_masking"]

bnb_config = None

if use_qlora:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        load_in_8bit=not quantization_config["load_in_4bit"],
    )

lora_config = LoraConfig(
    **lora_config,
    task_type="CAUSAL_LM",
)


def get_apply_peft(
    model_name: str,
    lora_config: LoraConfig,
    qlora_config: Optional[BitsAndBytesConfig] = None,
) -> torch.nn.Module:

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=qlora_config, device_map="auto"
    )

    return get_peft_model(model, lora_config)


# Load model without quantization for original parameter count
model = AutoModelForCausalLM.from_pretrained(model_name)
original_params = count_trainable_params(model)


peft_model = get_apply_peft(model_name, lora_config, bnb_config)

peft_params = count_trainable_params(peft_model)


print(f"Original params: {original_params}")
print(f"PEFT model params: {peft_params}")

print(
    f"Percentage of trainable params after PEFT: {peft_params / original_params * 100:.2f}%"
)


print(f"Original model size: {get_model_size_gb(model):.2f} GB")
print(f"PEFT model size: {get_model_size_gb(peft_model):.2f} GB")


tokenized_dataset, tokenizer = tokenize_dataset(
    model_name, assistant_only_masking=assistant_only_masking
)


# Setup data collator for padding
print("\n" + "=" * 50)
print("SETTING UP TRAINING")
print("=" * 50)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked LM
)

# Training arguments
training_args = TrainingArguments(**training_args)

# Print masking statistics if using assistant-only masking
if assistant_only_masking:
    first_example = tokenized_dataset[0]
    input_ids = first_example["input_ids"]
    labels = first_example["labels"]

    masked_tokens = sum(1 for label in labels if label == -100)
    total_tokens = len(labels)

    print(f"\nMasking statistics for first example:")
    print(f"- Total tokens: {total_tokens}")
    print(f"- Masked tokens: {masked_tokens}")
    print(f"- Training tokens: {total_tokens - masked_tokens}")
    print(f"- Masking ratio: {masked_tokens / total_tokens * 100:.1f}%")

# Setup SFTTrainer
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
