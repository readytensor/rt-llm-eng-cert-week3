import sys
import os
import torch
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from utils import (
    count_trainable_params,
    read_json_file,
    get_model_size_gb,
)
from prepare_dataset import tokenize_dataset
from paths import CONFIG_FILE

from typing import Optional
from huggingface_hub import login


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

config = read_json_file(CONFIG_FILE)

model_name = config["model_name"]
dataset_config = config["dataset_config"]
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


peft_model = get_apply_peft(model_name, lora_config, bnb_config)


print(f"PEFT model size: {get_model_size_gb(peft_model):.2f} GB")


train, validation, test, tokenizer = tokenize_dataset(
    model_name=model_name,
    assistant_only_masking=assistant_only_masking,
    **dataset_config,
)


# Setup data collator for padding
print("\n" + "=" * 50)
print("SETTING UP TRAINING")
print("=" * 50)


# Training arguments
training_args = TrainingArguments(**training_args)

# Print masking statistics if using assistant-only masking
if assistant_only_masking:
    first_example = train[0]
    input_ids = first_example["input_ids"]
    labels = first_example["labels"]

    masked_tokens = sum(1 for label in labels if label == -100)
    total_tokens = len(labels)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train,
    eval_dataset=validation,
)

trainer.train()
