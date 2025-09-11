import sys
import os
import torch
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import read_json_file
from paths import CONFIG_FILE


def load_trained_model(checkpoint_path: str, base_model_name: Optional[str] = None):
    """
    Load a fine-tuned model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        base_model_name: Base model name (if None, loads from config)

    Returns:
        tuple: (model, tokenizer)
    """
    # Load config if base model not specified
    if base_model_name is None:
        config = read_json_file(CONFIG_FILE)
        base_model_name = config["model_name"]

    print(f"Loading base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load PEFT model from checkpoint
    print(f"Loading PEFT adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Enable evaluation mode
    model.eval()

    return model, tokenizer


def test_model_with_question(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Test the model with a single question.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        question: Question to ask the model
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling

    Returns:
        Generated response
    """
    # Format the question using the same format as training
    test_prompt = f"### Question\n{question}\n\n### Output\n"

    print(f"Question: {question}")
    print(f"Prompt: {repr(test_prompt)}")

    # Tokenize the prompt
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (after the prompt)
    generated_text = full_response[len(test_prompt) :].strip()

    return generated_text


def test_model_interactive(checkpoint_path: str, base_model_name: Optional[str] = None):
    """
    Interactive testing session with the model.

    Args:
        checkpoint_path: Path to the checkpoint directory
        base_model_name: Base model name (optional)
    """
    print("=" * 60)
    print("LOADING FINE-TUNED MODEL FOR TESTING")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_trained_model(checkpoint_path, base_model_name)

    print("\nModel loaded successfully!")
    print("Enter questions to test the model (type 'quit' to exit)")
    print("-" * 60)

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            break

        if not question:
            continue

        print("\nGenerating response...")
        try:
            response = test_model_with_question(model, tokenizer, question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")

        print("-" * 60)

    print("Testing session ended.")


def test_model_batch(
    checkpoint_path: str, questions: list, base_model_name: Optional[str] = None
):
    """
    Test the model with a batch of questions.

    Args:
        checkpoint_path: Path to the checkpoint directory
        questions: List of questions to test
        base_model_name: Base model name (optional)

    Returns:
        List of responses
    """
    print("=" * 60)
    print("BATCH TESTING FINE-TUNED MODEL")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_trained_model(checkpoint_path, base_model_name)

    responses = []

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Testing question: {question}")

        try:
            response = test_model_with_question(model, tokenizer, question)
            responses.append(response)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            responses.append(f"ERROR: {e}")

        print("-" * 40)

    return responses


if __name__ == "__main__":

    test_model_interactive(
        "/Users/mo/Desktop/ReadyTensor/certifications/llm-finetuning/repos/rt-llm-finetuning-cert-week3/results/checkpoint-90",
        "meta-llama/Llama-3.2-1B-Instruct",
    )
