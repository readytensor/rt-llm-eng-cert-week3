![Supervised Fine-Tuning with TRL - Hero Image](sft-hero.webp)

--DIVIDER-----

---

[⬅️ Previous - TBD](TBD)
[➡️ Next - TBD](TBD)

---

# TL;DR

**Core Insight**: Supervised fine-tuning transforms pre-trained models into specialized assistants through careful data preparation and training
**Why It Matters**: SFT is the foundation for creating instruction-following models that can perform specific tasks effectively
**Key Process**: Dataset preparation → tokenization → assistant masking → padding → training with SFTTrainer
**Mental Model**: Think of SFT as teaching a knowledgeable student to follow specific instructions and respond in desired formats

# The Foundation: Supervised Fine-Tuning in Practice

Welcome to the hands-on implementation of supervised fine-tuning! In our previous lesson, we learned that language models are sophisticated classification systems. Now we'll put that understanding to work by building a complete fine-tuning pipeline.

This lesson will take you through the entire process of preparing data, configuring training, and creating a specialized model. Instead of just showing you the final code, we'll explore what's happening at each step and why these choices matter for successful fine-tuning.

## What You'll Learn This Lesson

By the end of this lesson, you'll have a deep understanding of:

- How to prepare and structure datasets for instruction following
- The tokenization process and why it's crucial for model training
- Assistant-only masking: ensuring models learn to respond, not repeat
- Padding strategies and data collation for efficient batch processing
- Training with SFTTrainer and the magic happening under the hood

Let's dive into the practical mechanics that make fine-tuning work! 🚀

## The Six Steps of Supervised Fine-Tuning

Before we dive into implementation details, let's understand the six essential steps that make supervised fine-tuning successful. Each step builds upon the previous ones, creating a robust foundation for training instruction-following models.

Think of these as the essential ingredients in a recipe - skip one, and your model won't learn effectively. Master all six, and you'll have the tools to fine-tune any language model for your specific needs.

### The Complete Pipeline Overview

Here's what we're building:
```
Raw Data → Dataset Preparation → Model Loading & LoRA Setup → Tokenization → Assistant Masking → Padding → Training
```

Each step transforms your data and configures your model to learn exactly what you want it to learn - no more, no less.

--DIVIDER--

## Step 1: Dataset Preparation - Structuring Data for Learning

The foundation of any successful fine-tuning project is a well-prepared dataset. This isn't just about collecting conversations - it's about structuring them in a way that clearly communicates to the model what behavior you want to teach.

### Understanding Conversation Structure

When we fine-tune a model, we're essentially showing it examples of how to behave. Each data point in your dataset becomes a template that the model learns to follow. Here's what a typical training sample looks like:

```python
# Raw data point
data_point = {
    "question": "What's the capital of France?",
    "input": "",  # Optional additional context
    "output": "The capital of France is Paris."
}
```

But the model doesn't understand structured data - it only understands text sequences. So we need to convert this structured data into a single text string that the model can process.

### The Instruction Format: Converting Structure to Text

This is where consistent formatting becomes crucial. We use a clear instruction format that helps the model understand the different parts of each example:

```python
def format_instruction_data(data_point):
    """Format a data point into instruction format for fine-tuning."""
    question = data_point["question"]
    input_text = data_point["input"]
    output = data_point["output"]
    
    formatted_text = f"### Question\n{question}\n\n"
    
    if input_text:
        formatted_text += f"### Input\n{input_text}\n\n"
    
    formatted_text += f"### Output\n{output}"
    
    return formatted_text

# Example output:
# ### Question
# What's the capital of France?
# 
# ### Output
# The capital of France is Paris.
```

This format uses clear section headers (`### Question`, `### Input`, `### Output`) that help the model understand the structure of instruction-following tasks.

### Preparing Your Dataset

Here's how to prepare a complete dataset for fine-tuning using our instruction format:

```python
import json
from datasets import Dataset
from tqdm import tqdm

def load_jsonl_dataset(file_path):
    """Load a JSONL dataset from file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def prepare_dataset(file_path):
    """Load and prepare the dataset for fine-tuning."""
    # Load raw data
    raw_data = load_jsonl_dataset(file_path)
    
    # Format each data point
    formatted_data = []
    for data_point in tqdm(raw_data, desc="Preparing dataset"):
        formatted_text = format_instruction_data(data_point)
        formatted_data.append({"text": formatted_text})
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)
    return dataset
```

The key insight here is that we're not just collecting data - we're teaching the model a specific instruction-following pattern. The model learns to recognize questions and generate appropriate responses in the `### Output` section.

### Why Format Matters

Using a consistent instruction format isn't optional - it's critical for several reasons:

**Clear Structure**: The section headers (`### Question`, `### Input`, `### Output`) create clear boundaries that help the model understand different parts of the instruction-following task.

**Learning Focus**: By consistently placing the desired response in the `### Output` section, we train the model to generate responses only when it sees this pattern.

**Instruction Following**: The formatting teaches the model to read the question, consider any additional input, and then provide a focused response in the output section.

--DIVIDER--

## Step 2: Model Loading and LoRA Setup - Preparing for Efficient Training

Before we can tokenize our data, we need to load our base model and configure it for parameter-efficient fine-tuning. This step is crucial because it determines how much memory we'll use and how efficiently we can train.

### Choosing Between LoRA and QLoRA

Before loading the model, you need to decide between two parameter-efficient approaches:

- **LoRA**: Uses the full-precision model (16-bit) with adapter layers
- **QLoRA**: Quantizes the base model to 4-bit and adds 16-bit adapter layers

The choice depends on your available GPU memory:

### Option 1: LoRA (More Memory, Potentially Better Quality)

For LoRA, load the model in full precision:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load model in full precision for LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### Option 2: QLoRA (Less Memory, Still Great Results)

For QLoRA, use 4-bit quantization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Configure 4-bit quantization for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### Setting Up the Adapter Configuration

Regardless of whether you chose LoRA or QLoRA, the adapter configuration is the same:

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=8,                            # Rank of adaptation
    lora_alpha=32,                  # Scaling parameter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,              # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 345,059,328 || trainable%: 0.085
```

### Why Choose LoRA vs QLoRA?

Each approach has distinct advantages:

**LoRA Advantages**:
- Full precision base model may give slightly better quality
- Simpler setup (no quantization configuration needed)
- Faster inference since no dequantization required

**QLoRA Advantages**:
- Dramatically reduced memory usage (~75% less GPU memory)
- Makes large model fine-tuning possible on consumer GPUs
- Nearly identical performance to LoRA in most cases

**Both Approaches Share**:
- Tiny adapter weights that are easy to share and store
- Much faster training than full fine-tuning
- Ability to swap different adapters with the same base model

For this tutorial, we'll use **LoRA** (the simpler option), but you can easily switch to QLoRA by using the quantization configuration shown above.

--DIVIDER--

## Step 3: Tokenization - Converting Text to Numbers

Once we have properly formatted text, we need to convert it into numbers that the model can actually process. This is where tokenization comes in - the bridge between human-readable text and machine-readable numbers.

### Understanding the Tokenization Process

Tokenization breaks text into smaller pieces called tokens, then converts each token to a unique number. Here's what this looks like in practice:

```python
# Take our formatted instruction text
text = "### Question\nWhat's the capital of France?\n\n### Output\nThe capital of France is Paris."

# Tokenize it
tokens = tokenizer(text)
print("Token IDs:", tokens["input_ids"][:10])  # First 10 tokens
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][:10]))

# Output might look like:
# Token IDs: [2, 1, 1, 894, 18233, 198, 3923, 596, 279, 6864]
# Tokens: ['<s>', '##', '#', 'Question', '\n', 'What', "'s", ' the', ' capital']
```

Notice how the tokenizer handles different elements:
- Section headers like `### Question` get broken into multiple tokens
- Words like "What's" get split into multiple tokens
- Newlines and spaces become separate tokens

### The Vocabulary: Model's Dictionary

Every model has a fixed vocabulary - typically around 50,000 tokens. This vocabulary was determined during the model's original training and cannot be changed. The tokenizer's job is to break any text into pieces that exist in this vocabulary:

```python
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens_map}")

# Example of how unknown words get handled
text_with_rare_word = "The quixotic adventure began"
tokens = tokenizer.tokenize(text_with_rare_word)
print(f"Tokens: {tokens}")
# Output: ['The', ' qu', 'ix', 'otic', ' adventure', ' began']
```

Even rare words like "quixotic" get broken down into subword pieces that the model recognizes. This subword tokenization is what allows models to handle any text, even words they've never seen before.

### Tokenization in the Training Pipeline

When preparing data for training, tokenization happens as part of the dataset preparation. Here's how our actual implementation works:

```python
def tokenize_and_mask_function(examples):
    # Add EOS token to each text
    texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts_with_eos,
        truncation=True,
        padding=False,  # We'll handle padding during training
        max_length=512,
        return_tensors=None,
        add_special_tokens=True,
    )
    
    # We'll add masking logic here (covered in next section)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply to your dataset
tokenized_dataset = dataset.map(tokenize_and_mask_function, batched=True)
```

The key parameters here are:
- **truncation=True**: Cut off sequences longer than max_length
- **padding=False**: Don't add padding tokens yet (we'll do this during training)
- **max_length=512**: Maximum sequence length for our examples
- **add_special_tokens=True**: Add beginning-of-sequence tokens

### Why Tokenization Details Matter

Understanding tokenization helps you avoid common pitfalls:

**Sequence Length Planning**: Knowing how many tokens your conversations become helps you set appropriate max_length values. A conversation that looks short in text might become hundreds of tokens.

**Special Token Handling**: Different models use different special tokens. Using the wrong tokenizer can completely break your fine-tuning.

**Efficiency Considerations**: Longer sequences require more memory and compute. Understanding token counts helps you optimize your training setup.

--DIVIDER--

## Step 4: Assistant-Only Masking - Teaching What to Learn

Here's where supervised fine-tuning gets sophisticated. We don't want our model to learn to generate the entire conversation - we only want it to learn to generate the assistant's responses. This is where assistant-only masking becomes crucial.

### The Problem: Learning the Wrong Thing

Without proper masking, the model would try to learn to predict every token in the sequence, including the question and input sections. This creates several problems:

```python
# Without masking, the model learns to predict:
# "### Question\nWhat's the" → "capital of France?"  (Not what we want!)
# "### Output\nThe capital" → "of France is Paris."  (This is what we want)
```

We want the model to learn to generate responses, not memorize questions. Assistant-only masking solves this by telling the model which tokens to learn from and which to ignore.

### How Masking Works

Masking uses a special value (-100) in the labels to tell the training process to ignore certain tokens when calculating loss:

```python
def apply_assistant_masking(input_ids, tokenizer):
    """Apply assistant-only masking by setting instruction tokens to -100."""
    labels = input_ids.copy()
    
    # Find the "### Output" marker to identify where assistant response starts
    output_marker = "### Output"
    output_marker_tokens = tokenizer.encode(output_marker, add_special_tokens=False)
    
    # Find where the output section begins
    output_start_idx = None
    for i in range(len(input_ids) - len(output_marker_tokens) + 1):
        if input_ids[i : i + len(output_marker_tokens)] == output_marker_tokens:
            output_start_idx = i + len(output_marker_tokens)
            break
    
    # If we found the output marker, mask everything before it
    if output_start_idx is not None:
        # Mask instruction tokens (set to -100)
        for i in range(output_start_idx):
            labels[i] = -100
    
    return labels
```

When the training process sees -100 in the labels, it skips those positions when calculating the loss. This means the model only learns from tokens in the `### Output` section.

### Integrating Masking into Dataset Preparation

In our implementation, we apply masking during the dataset tokenization process:

```python
def tokenize_and_mask_function(examples):
    # Add EOS token and tokenize
    texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
    tokenized = tokenizer(texts_with_eos, truncation=True, padding=False, max_length=512)
    
    # Apply assistant-only masking to each example
    labels = []
    for input_ids in tokenized["input_ids"]:
        if assistant_only_masking:
            masked_labels = apply_assistant_masking(input_ids, tokenizer)
        else:
            masked_labels = input_ids
        labels.append(masked_labels)
    
    tokenized["labels"] = labels
    return tokenized

# Apply to dataset
tokenized_dataset = dataset.map(tokenize_and_mask_function, batched=True)
```

This approach:
1. Tokenizes the instruction-formatted text
2. Finds the `### Output` marker in each example
3. Masks all tokens before the output section with -100
4. Ensures the model only learns from the response portion

### Why This Matters for Model Behavior

Proper masking is what makes the difference between a model that:
- **With masking**: Reads questions and generates focused responses in the output format
- **Without masking**: Might generate questions, repeat input text, or produce unfocused responses

This is why fine-tuned models can follow instructions effectively rather than just continuing text in any direction.

--DIVIDER--

## Step 5: Padding and Data Collation - Efficient Batch Processing

Training neural networks requires processing multiple examples simultaneously in batches. But conversations have different lengths, and models need fixed-size inputs. This is where padding and data collation become essential.

### The Batching Challenge

Consider these three conversations:
```
Conversation 1: 45 tokens
Conversation 2: 123 tokens  
Conversation 3: 67 tokens
```

To process them together, we need to make them all the same length. Padding solves this by adding special padding tokens to shorter sequences:

```python
# Before padding (different lengths)
batch = [
    [1, 2, 3, 4, 5],           # 5 tokens
    [1, 2, 3],                 # 3 tokens  
    [1, 2, 3, 4, 5, 6, 7, 8]   # 8 tokens
]

# After padding (all length 8)
padded_batch = [
    [1, 2, 3, 4, 5, 0, 0, 0],     # Padded with 0s
    [1, 2, 3, 0, 0, 0, 0, 0],     # Padded with 0s
    [1, 2, 3, 4, 5, 6, 7, 8]      # No padding needed
]
```

### Smart Padding with Data Collators

Instead of padding all sequences to the maximum possible length (which wastes memory), we can pad each batch to the length of the longest sequence in that batch:

```python
from transformers import DataCollatorForLanguageModeling

# Create a data collator that handles padding dynamically
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked LM
    pad_to_multiple_of=8,  # Pad to multiples of 8 for efficiency
)

# The collator automatically pads each batch appropriately
```

This approach is much more memory-efficient because:
- Batch 1 might be padded to length 120
- Batch 2 might be padded to length 89
- Batch 3 might be padded to length 156

Each batch only uses the memory it actually needs.

### Attention Masks: Ignoring Padding

When we add padding tokens, we need to tell the model to ignore them. This is done through attention masks:

```python
# Example of how attention masks work
sequence = [1, 2, 3, 0, 0]  # 0 = padding token
attention_mask = [1, 1, 1, 0, 0]  # 1 = attend, 0 = ignore

# The model only pays attention to the real tokens (1, 2, 3)
# and ignores the padding tokens (0, 0)
```

### Integration with SFTTrainer

SFTTrainer handles all of this automatically:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Handles padding
    max_seq_length=1024,          # Maximum sequence length
)
```


### Why Efficient Batching Matters

Proper padding and collation affect:
- **Training Speed**: Less padding means faster training
- **Memory Usage**: Efficient batching lets you use larger batch sizes
- **Model Quality**: Better GPU utilization means more training steps per hour

These optimizations can reduce training time from hours to minutes for the same dataset.

--DIVIDER--

## Step 6: Training with SFTTrainer - Bringing It All Together

Now we reach the culmination of our preparation work. SFTTrainer coordinates all the pieces we've built - the formatted dataset, tokenization, masking, and padding - into a seamless training process.

### The Magic of SFTTrainer

What makes SFTTrainer special is how it handles the complexity of instruction fine-tuning automatically:

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

# Configure LoRA for efficient training
lora_config = LoraConfig(
    r=64,                    # Rank of adaptation
    lora_alpha=16,           # Scaling parameter
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

# Set up training parameters
training_args = TrainingArguments(
    output_dir="./sft-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=1000,
    logging_steps=10,
    save_steps=200,
    optim="paged_adamw_8bit",
    fp16=True,
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    args=training_args,
    max_seq_length=1024,
    packing=True,
)
```

### What Happens During Training

When you call `trainer.train()`, here's what happens behind the scenes:

**Step 1: Batch Creation**
- SFTTrainer loads a batch of conversations from your dataset
- Processes instruction-formatted text automatically
- Tokenizes the text using your specified tokenizer

**Step 2: Dynamic Padding**
- Pads sequences in the batch to the same length
- Creates attention masks to ignore padding tokens
- Applies assistant-only masking to focus learning

**Step 3: Forward Pass**
- Feeds the batch through your model
- Model predicts next tokens for each position
- Calculates probabilities for all vocabulary tokens

**Step 4: Loss Calculation**
- Compares predictions to actual next tokens
- Only calculates loss for assistant response tokens (thanks to masking)
- Ignores padding tokens and user input tokens

**Step 5: Backward Pass**
- Calculates gradients based on the loss
- Updates only LoRA adapter weights (not the full model)
- Applies gradient accumulation if specified

### The Training Loop in Action

```python
# Start training
trainer.train()

# What you'll see:
# Step 10: loss=2.45, learning_rate=0.0002
# Step 20: loss=2.12, learning_rate=0.00019
# Step 30: loss=1.89, learning_rate=0.00018
# ...
```

The decreasing loss indicates that your model is learning to better predict assistant responses. A typical fine-tuning run might see loss drop from 2.5 to 1.2 over 1000 steps.

### Memory Efficiency in Practice

SFTTrainer with LoRA makes efficient use of your GPU memory:

```python
# Without LoRA: Need ~40GB for LLaMA-7B fine-tuning
# With LoRA + 4-bit quantization: Need ~6GB for the same model

# This is achieved through:
# - Quantizing the base model to 4-bit precision
# - Only training small LoRA adapter matrices
# - Dynamic padding to minimize wasted memory
# - Gradient checkpointing to trade compute for memory
```

### Monitoring Training Progress

SFTTrainer integrates with popular monitoring tools:

```python
training_args = TrainingArguments(
    # ... other args ...
    report_to="tensorboard",    # or "wandb"
    logging_dir="./logs",
    logging_steps=10,
)

# View progress in TensorBoard
# tensorboard --logdir ./logs
```

You can monitor:
- Training loss (should decrease steadily)
- Learning rate schedule
- Memory usage
- Training speed (tokens per second)

### Saving and Sharing Your Model

After training completes, save your results:

```python
# Save the LoRA adapters
trainer.model.save_pretrained("./my-sft-model")
tokenizer.save_pretrained("./my-sft-model")

# Push to Hugging Face Hub for sharing
trainer.push_to_hub("my-username/my-sft-model")

# Load for inference later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my-sft-model")
```

The beauty of LoRA is that your adapters are tiny (typically <100MB) compared to the full model (several GB), making sharing and deployment much easier.

--DIVIDER--

## 🎥 Video: Watching SFT Training in Real-Time

*[VIDEO PLACEHOLDER - This section will contain an interactive demonstration showing:]*

*- A complete SFT training run from start to finish*
*- Real-time loss curves and how they indicate learning progress*
*- Memory usage patterns during training with LoRA vs full fine-tuning*
*- Before/after model behavior comparisons*
*- Common training issues and how to diagnose them*

*The video will make the abstract training process concrete by showing actual numbers, graphs, and model outputs.*

--DIVIDER--

## From Theory to Practice: The Complete SFT Pipeline

### Understanding the Full Workflow

Now that we've explored each pillar individually, let's see how they work together in a complete fine-tuning pipeline. This integration is where the magic happens - each component enhances the others to create an efficient and effective training process.

Think of it like cooking a complex dish. You can understand each ingredient and technique separately, but the real art is in how they combine. The timing of when you add each component, how they interact, and the final result that emerges from their coordination.

### The Synergy Between Components

**Dataset Preparation + Tokenization**: Proper instruction formatting ensures that tokenization preserves the question-answer structure. The section headers (`### Question`, `### Output`) become crucial markers that help the model understand instruction-following patterns.

**Tokenization + Masking**: The tokenizer's processing of section headers enables precise masking. We can identify exactly where the `### Output` section begins, creating clean learning boundaries between questions and responses.

**Masking + Padding**: Assistant-only masking ensures we only learn from response tokens, while efficient padding maximizes the learning signal per batch. Together, they optimize both quality and efficiency.

**Padding + Training**: Dynamic batching with SFTTrainer means we're constantly optimizing memory usage while maintaining the integrity of our learning objectives.

### Real-World Performance Implications

This integrated approach delivers measurable benefits:

**Training Speed**: Proper packing and padding can reduce training time by 2-3x compared to naive approaches.

**Memory Efficiency**: LoRA + quantization + efficient batching lets you fine-tune 7B parameter models on consumer GPUs that would otherwise require enterprise hardware.

**Model Quality**: Assistant-only masking produces models that follow instructions cleanly rather than generating both sides of conversations or repeating user inputs.

**Reproducibility**: The systematic approach ensures that your results are consistent and can be replicated by others.

--DIVIDER--

## Key Takeaways

- **Dataset preparation sets the foundation** - proper instruction formatting and structure determine what your model learns
- **Choose between LoRA and QLoRA based on your GPU memory** - LoRA for simplicity, QLoRA for maximum memory efficiency
- **Tokenization bridges human and machine understanding** - the right tokenizer and format are crucial for effective learning  
- **Assistant-only masking focuses learning** - models learn to respond appropriately rather than memorize entire conversations
- **Efficient padding maximizes resources** - smart batching dramatically improves training efficiency
- **SFTTrainer orchestrates everything** - it coordinates all components into a seamless, optimized training process

Understanding these six steps gives you the foundation to fine-tune any language model effectively. Each component serves a specific purpose, but their real power emerges when they work together in harmony.

In our next lesson, we'll explore advanced techniques that build on this foundation: parameter-efficient fine-tuning methods, training optimization strategies, and how to evaluate your fine-tuned models effectively.

---

[⬅️ Previous - TBD](TBD)
[➡️ Next - TBD](TBD)

---
