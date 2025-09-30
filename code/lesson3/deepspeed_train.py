import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.distributed as dist


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from utils import read_json_file, push_to_hub
from paths import (
    CONFIG_FILE,
    DEEP_SPEED_ZERO1_CONFIG,
    DEEP_SPEED_ZERO2_CONFIG,
    DEEP_SPEED_ZERO3_CONFIG,
)
from lesson1.prepare_dataset import tokenize_dataset, DataCollatorForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
login(HF_TOKEN)


def main(
    model_id: str,
    lora_config: dict,
    dataset_config: dict,
    training_args: dict,
    deepspeed_version: int,
    save_model_name: str,
):
    # Paths and configuration
    output_dir = f"./qlora-deepspeed-zero{deepspeed_version}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting training with model: {model_id}")

    assert deepspeed_version in [1, 2, 3], "DeepSpeed version must be 1, 2, or 3"

    if deepspeed_version == 1:
        ds_config_path = DEEP_SPEED_ZERO1_CONFIG
    elif deepspeed_version == 2:
        ds_config_path = DEEP_SPEED_ZERO2_CONFIG
    elif deepspeed_version == 3:
        ds_config_path = DEEP_SPEED_ZERO3_CONFIG

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config
    )

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Set up LoRA configuration
    peft_config = LoraConfig(
        **lora_config,
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    print("LoRA applied")

    # Load dataset
    print("Loading and processing dataset...")
    train, validation, test, tokenizer = tokenize_dataset(
        model_id,
        assistant_only_masking=True,
        **dataset_config,
    )
    print(
        f"Dataset loaded - Train: {len(train)}, Val: {len(validation) if validation else 0}"
    )
    training_args = TrainingArguments(
        deepspeed=ds_config_path,
        **training_args,
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )

    trainer.train()

    # Save the fine-tuned model
    print("saving model...")
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")

    print(f"Training complete! Model saved to {output_dir}/final-model")

    model_name = f"{save_model_name}-deepspeed-zero{deepspeed_version}"

    if is_main_process():
        push_to_hub(model, tokenizer, model_name, HF_USERNAME)


if __name__ == "__main__":
    config = read_json_file(CONFIG_FILE)
    model_id = config["model_name"]
    lora_config = config["lora_config"]
    dataset_config = config["dataset_config"]
    training_args = config["training_args"]
    deepspeed_version = config["deepspeed_version"]
    save_model_name = config["save_model_name"]
    main(
        model_id,
        lora_config,
        dataset_config,
        training_args,
        deepspeed_version,
        save_model_name,
    )
