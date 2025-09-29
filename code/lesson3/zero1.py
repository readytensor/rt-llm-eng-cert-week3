import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from utils import read_json_file
from paths import CONFIG_FILE, DEEP_SPEED_ZERO1_CONFIG
from lesson1.prepare_dataset import tokenize_dataset, DataCollatorForCausalLM


def main(model_id: str, lora_config: dict, dataset_config: dict, training_args: dict):
    # Paths and configuration
    output_dir = "./qlora-deepspeed-zero1"

    # Write the config to a file
    ds_config_path = DEEP_SPEED_ZERO1_CONFIG

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
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

    train, validation, test, tokenizer = tokenize_dataset(
        model_id,
        assistant_only_masking=True,
        **dataset_config,
    )
    # Define training arguments with DeepSpeed integration
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

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")

    print(f"Training complete! Model saved to {output_dir}/final-model")


if __name__ == "__main__":
    config = read_json_file(CONFIG_FILE)
    model_id = config["model_name"]
    lora_config = config["lora_config"]
    dataset_config = config["dataset_config"]
    training_args = config["training_args"]
    main(model_id, lora_config, dataset_config, training_args)
