import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from paths import DEEP_SPEED_ZERO2_CONFIG


def main():
    # Paths and configuration
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "./llama-qlora-deepspeed-zero2"

    # open zero2.json
    ds_config_path = DEEP_SPEED_ZERO2_CONFIG

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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Load and prepare dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    print(f"Available dataset splits: {dataset.keys()}")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments with DeepSpeed integration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        per_device_train_batch_size=2,  # Explicit batch size per device
        gradient_accumulation_steps=4,
        deepspeed=ds_config_path,
    )

    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")

    print(f"Training complete! Model saved to {output_dir}/final-model")


if __name__ == "__main__":
    main()
