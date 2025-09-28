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


def interactive_testing(ft_model, ft_tokenizer, base_model, base_tokenizer):
    """Interactive testing mode - compare fine-tuned vs base model responses"""

    print("\n" + "=" * 80)
    print("🎯 INTERACTIVE COMPARISON MODE")
    print("=" * 80)
    print("Ask questions and see responses from BOTH models:")
    print("  🤖 Fine-tuned model (your trained model)")
    print("  🔧 Base model (original Llama)")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit', 'exit', or 'q' to stop")
    print("  - Type 'help' for sample questions")
    print("  - Type 'clear' to clear the screen")
    print("-" * 80)

    sample_questions = [
        "What are the key elements of a non-disclosure agreement?",
        "Explain the difference between a contract and an agreement.",
        "What should be included in a software licensing agreement?",
        "Draft a simple liability clause for a consulting contract.",
        "What are the essential clauses in an employment agreement?",
        "How do I protect intellectual property in a partnership agreement?",
        "What are the key terms in a service level agreement?",
        "Explain force majeure clauses in contracts.",
    ]

    while True:
        try:
            # Get user input
            print("\n" + "=" * 50)
            user_input = input("🤔 Your question: ").strip()

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Thanks for testing! Goodbye!")
                break

            elif user_input.lower() == "help":
                print("\n📝 Sample questions you can ask:")
                for i, question in enumerate(sample_questions, 1):
                    print(f"  {i}. {question}")
                continue

            elif user_input.lower() == "clear":
                import os

                os.system("cls" if os.name == "nt" else "clear")
                print("🎯 INTERACTIVE TESTING MODE")
                print("Ask your fine-tuned model questions about legal topics!")
                continue

            elif not user_input:
                print("❓ Please enter a question or type 'help' for examples.")
                continue

            # Check if user wants to add context/input
            context_input = input(
                "📄 Additional context (optional, press Enter to skip): "
            ).strip()

            # Format the prompt
            if context_input:
                prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_input}

{context_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

            # Generate responses from both models
            print("\n🤖 Generating responses from both models...")
            print("=" * 80)

            try:
                # Fine-tuned model response
                print("🤖 FINE-TUNED MODEL RESPONSE:")
                print("-" * 40)
                ft_response = generate_response(
                    ft_model, ft_tokenizer, prompt, max_new_tokens=300
                )
                print(ft_response)

                print("\n" + "=" * 80)

                # Base model response
                print("🔧 BASE MODEL RESPONSE:")
                print("-" * 40)
                base_response = generate_response(
                    base_model, base_tokenizer, prompt, max_new_tokens=300
                )
                print(base_response)

                # Ask if user wants to continue
                print("\n" + "=" * 80)
                continue_choice = (
                    input("❓ Ask another question? (y/n/help): ").strip().lower()
                )

                if continue_choice in ["n", "no"]:
                    print("👋 Thanks for testing! Goodbye!")
                    break
                elif continue_choice == "help":
                    print("\n📝 Sample questions you can ask:")
                    for i, question in enumerate(sample_questions, 1):
                        print(f"  {i}. {question}")

            except Exception as e:
                print(f"❌ Error generating response: {e}")
                print("🔄 Please try again with a different question.")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🔄 Please try again.")


if __name__ == "__main__":
    print("🚀 Loading models for comparison...")
    print(f"Base model: {base_model_name}")
    print(f"Fine-tuned adapter: {adapter_model_name}")

    try:
        # Load the fine-tuned model
        print("\n📥 Loading fine-tuned model...")
        ft_model, ft_tokenizer = load_fine_tuned_model(
            base_model_name, adapter_model_name
        )

        # Load the base model
        print("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        print("✅ Both models loaded successfully!")

        # Start interactive comparison
        interactive_testing(ft_model, ft_tokenizer, base_model, base_tokenizer)

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nPossible solutions:")
        print("1. Make sure your model is pushed to Hugging Face Hub")
        print("2. Check your HF_USERNAME and save_model_name in config")
        print("3. Verify you have access to the model repository")
        print("4. Try running: huggingface-cli login")
