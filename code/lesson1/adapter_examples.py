"""
Examples of different ways to load and use LoRA adapters
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


# Method 1: Load base model + adapters separately (most common)
def load_method_1():
    print("Method 1: Base model + adapters")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, "your-username/llama-1b-legal-lora")
    tokenizer = AutoTokenizer.from_pretrained("your-username/llama-1b-legal-lora")

    return model, tokenizer


# Method 2: Auto load everything (simpler)
def load_method_2():
    print("Method 2: Auto load everything")

    model = AutoPeftModelForCausalLM.from_pretrained(
        "your-username/llama-1b-legal-lora",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained("your-username/llama-1b-legal-lora")

    return model, tokenizer


# Method 3: Load from local path
def load_method_3():
    print("Method 3: Load from local path")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, "./final_adapters")
    tokenizer = AutoTokenizer.from_pretrained("./final_adapters")

    return model, tokenizer


# Method 4: Merge adapters into base model (for deployment)
def merge_and_save():
    print("Method 4: Merge adapters into base model")

    # Load model with adapters
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, "your-username/llama-1b-legal-lora")

    # Merge adapters into base model
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained("./merged_model")

    # Now you can load it as a regular model
    tokenizer = AutoTokenizer.from_pretrained("your-username/llama-1b-legal-lora")
    tokenizer.save_pretrained("./merged_model")

    print("Merged model saved to ./merged_model")
    return merged_model, tokenizer


if __name__ == "__main__":
    print("Choose a method to load your fine-tuned model:")
    print("1. Base model + adapters (recommended)")
    print("2. Auto load everything")
    print("3. Load from local path")
    print("4. Merge adapters (for deployment)")

    # Example: Use method 1
    # model, tokenizer = load_method_1()
    # print("Model loaded successfully!")
