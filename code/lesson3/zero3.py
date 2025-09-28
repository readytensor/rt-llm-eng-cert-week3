import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def main():
    # Paths and configuration
    model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Consider using a smaller model if available
    output_dir = "./llama-qlora-deepspeed-zero3"
    
    # Create a simpler DeepSpeed config file with ZeRO-2 (more stable than ZeRO-3)

    
    # Write the config to a file
    ds_config_path = os.path.join(os.getcwd(), "zero3.json")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"  # Let the library manage device placement
    )
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA configuration with smaller rank
    peft_config = LoraConfig(
        r=8,  # Reduced from 16
        lora_alpha=16,  # Reduced from 32
        target_modules=["q_proj", "v_proj"],  # Reduced target modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    
    # Load a smaller dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    print(f"Available dataset splits: {dataset.keys()}")
    
    # Use a much smaller subset for testing
    
    # Tokenize the dataset with smaller sequence length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256  # Reduced from 512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Define training arguments with simpler DeepSpeed integration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=1e-4,  # Reduced learning rate
        fp16=True,
        logging_steps=5,
        save_steps=100,
        per_device_train_batch_size=1,  # Minimum batch size
        gradient_accumulation_steps=4,
        deepspeed=ds_config_path,
        gradient_checkpointing=True,
        max_grad_norm=0.5,  # Lower gradient clipping
        # Disable evaluation to save memory
        eval_strategy="no",
        # Save disk space
        save_total_limit=2,
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