"""
Script to load and use fine-tuned LoRA adapters from Hugging Face
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)


def load_fine_tuned_model(base_model_name: str, adapter_model_name: str):
    """
    Load a fine-tuned model with LoRA adapters

    Args:
        base_model_name: Original model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        adapter_model_name: Your adapter model name (e.g., "your-username/llama-1b-legal-lora")
    """
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)

    print("✅ Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate a response using the fine-tuned model"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the input prompt from response
    response = response[len(prompt) :].strip()

    return response


if __name__ == "__main__":
    # Configuration
    BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    ADAPTER_MODEL = (
        "your-username/llama-1b-legal-lora"  # Change this to your actual model name
    )

    # Load the fine-tuned model
    model, tokenizer = load_fine_tuned_model(BASE_MODEL, ADAPTER_MODEL)

    # Test the model
    test_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What are the key elements of a non-disclosure agreement?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print("\n" + "=" * 50)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 50)
    print(f"Prompt: {test_prompt}")
    print("\nResponse:")

    response = generate_response(model, tokenizer, test_prompt)
    print(response)
