"""
Script to load and test fine-tuned LoRA adapters from Hugging Face Hub
"""

import sys
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_json_file
from paths import CONFIG_FILE

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
login(HF_TOKEN)

# Load config
config = read_json_file(CONFIG_FILE)
base_model_name = config["model_name"]
save_model_name = config["save_model_name"]
adapter_model_name = f"{HF_USERNAME}/{save_model_name}"


def load_fine_tuned_model(base_model_name: str, adapter_model_name: str):
    """
    Load a fine-tuned model with LoRA adapters from Hugging Face Hub

    Args:
        base_model_name: Original model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        adapter_model_name: Your adapter model name (e.g., "username/model-name")
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading tokenizer from: {adapter_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

    print(f"Loading LoRA adapters from: {adapter_model_name}")
    model = PeftModel.from_pretrained(base_model, adapter_model_name)

    print("✅ Fine-tuned model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate a response using the fine-tuned model with optimized parameters"""

    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # Match training max_length
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate response with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # Ensure minimum response length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            use_cache=True,
        )

    # Decode only the new tokens (response part)
    input_length = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    return response


def test_model_with_samples(model, tokenizer):
    """Test the model with various legal questions"""

    test_cases = [
        {
            "instruction": "What are the key elements of a non-disclosure agreement?",
            "input": "",
        },
        {
            "instruction": "Explain the difference between a contract and an agreement.",
            "input": "",
        },
        {
            "instruction": "What should be included in a software licensing agreement?",
            "input": "For a SaaS application",
        },
        {
            "instruction": "Draft a simple liability clause.",
            "input": "For a consulting services contract",
        },
    ]

    print("\n" + "=" * 80)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        instruction = test_case["instruction"]
        input_text = test_case["input"]

        # Format the prompt using the same template as training
        if input_text:
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        print(f"\n🔍 Test Case {i}:")
        print(f"Instruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")
        print("-" * 50)

        try:
            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")

        print("-" * 80)


def compare_with_base_model(base_model_name: str, adapter_model_name: str):
    """Compare responses from base model vs fine-tuned model"""

    print("\n" + "=" * 80)
    print("COMPARING BASE MODEL VS FINE-TUNED MODEL")
    print("=" * 80)

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_fine_tuned_model(base_model_name, adapter_model_name)

    # Test prompt
    test_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What are the essential clauses in a service agreement?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    print(f"\nTest Question: What are the essential clauses in a service agreement?")
    print("\n" + "=" * 40)
    print("BASE MODEL RESPONSE:")
    print("=" * 40)

    try:
        base_response = generate_response(
            base_model, base_tokenizer, test_prompt, max_new_tokens=200
        )
        print(base_response)
    except Exception as e:
        print(f"❌ Error with base model: {e}")

    print("\n" + "=" * 40)
    print("FINE-TUNED MODEL RESPONSE:")
    print("=" * 40)

    try:
        ft_response = generate_response(
            ft_model, ft_tokenizer, test_prompt, max_new_tokens=200
        )
        print(ft_response)
    except Exception as e:
        print(f"❌ Error with fine-tuned model: {e}")


if __name__ == "__main__":
    print("🚀 Loading fine-tuned model from Hugging Face Hub...")
    print(f"Base model: {base_model_name}")
    print(f"Adapter model: {adapter_model_name}")

    try:
        # Load the fine-tuned model
        model, tokenizer = load_fine_tuned_model(base_model_name, adapter_model_name)

        # Test with sample questions
        test_model_with_samples(model, tokenizer)

        # Optional: Compare with base model (uncomment if you want to see the difference)
        # compare_with_base_model(base_model_name, adapter_model_name)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPossible solutions:")
        print("1. Make sure your model is pushed to Hugging Face Hub")
        print("2. Check your HF_USERNAME and save_model_name in config")
        print("3. Verify you have access to the model repository")
        print("4. Try running: huggingface-cli login")
