![Supervised Fine-Tuning with TRL - Hero Image](sft-hero.webp)

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

Imagine you're teaching a brilliant student who already knows vast amounts of information but doesn't quite understand how to follow instructions or respond appropriately in conversations. That's essentially what we're doing with supervised fine-tuning - taking a pre-trained language model and teaching it the art of instruction-following through carefully crafted examples.

In our previous lesson, we discovered that language models are sophisticated classification systems, predicting the next most likely token in a sequence. Now we're going to harness that prediction power and channel it into something incredibly useful: a model that can understand instructions, process context, and generate helpful responses.

The journey we're about to embark on mirrors how human learning works. Just as you might learn a new skill by studying examples, practicing with guidance, and gradually improving through feedback, our model will learn from thousands of instruction-response pairs, each one teaching it a little more about how to be a helpful assistant.

What makes this process fascinating is that we're not just throwing data at the model and hoping for the best. Every decision we make - from how we format our training examples to which parts of the conversation we focus the model's attention on - has been carefully designed to maximize learning efficiency while minimizing computational cost. This is the art and science of supervised fine-tuning.

## 🎥 Video: Complete Supervised Fine-Tuning Walkthrough

*[VIDEO PLACEHOLDER - This section will contain a comprehensive demonstration showing:]*

*- Live coding of the complete fine-tuning pipeline from start to finish*
*- Real-time explanation of each configuration decision and its impact*
*- Watching training progress with loss curves and validation metrics*
*- Before/after model behavior comparisons with actual examples*
*- Common troubleshooting scenarios and debugging techniques*
*- Performance optimization tips for different hardware setups*

*The video will make the abstract concepts concrete by showing the actual implementation in action, complete with real data, training logs, and model outputs.*

## The Six Steps of Supervised Fine-Tuning

Think of supervised fine-tuning as preparing a gourmet meal. You wouldn't just throw random ingredients into a pot and hope for the best. Instead, you'd carefully select your ingredients, prepare them properly, combine them in the right order, and cook them with precise timing and temperature control. 

Our fine-tuning process follows the same philosophy. We have six essential steps, each one building upon the previous ones like layers in a carefully constructed dish. Miss one step or rush through it carelessly, and the entire outcome suffers. But execute each step with understanding and precision, and you'll create something remarkable: a specialized AI assistant that can follow instructions with impressive accuracy.

### The Complete Pipeline Overview

Here's what we're building:
![sft_flow.png](sft_flow.png)

This pipeline isn't just a technical process - it's a transformation journey. Raw conversational data enters on the left as unstructured text, and emerges on the right as a fine-tuned model that understands instructions, context, and appropriate responses. Each step along the way serves a specific purpose in this transformation, refining and shaping the data until it becomes the perfect teaching material for our model.

----
## Step 1: Dataset Preparation - The Art of Teaching Through Examples

Imagine you're a teacher preparing lesson materials for a new student. You wouldn't just hand them a pile of random textbooks and say "figure it out." Instead, you'd carefully curate examples that demonstrate exactly the skills and behaviors you want them to learn. You'd organize these examples in a clear, consistent format that makes the learning objectives obvious.

This is precisely what we're doing in our first step. We're not just loading data - we're crafting a curriculum. Every conversation in our dataset becomes a lesson that teaches our model how to understand instructions and generate appropriate responses. The key insight here is that the model learns not just from the content of these conversations, but from their structure and format.

Our implementation embraces a philosophy of configuration-driven development. Rather than hardcoding specific dataset paths or column names into our code, we externalize these decisions into a configuration file. This might seem like a small detail, but it represents a fundamental shift in how we approach machine learning experiments.

```python
# config.json
{
    "dataset_config": {
        "dataset_name": "NebulaSense/Legal_Clause_Instructions",
        "instruction_column": "Instruction", 
        "input_column": "Input",
        "output_column": "Output",
        "max_length": 2048
    }
}
```

Why does this matter? Because experimentation is at the heart of successful fine-tuning. You might want to try different datasets, compare how various instruction formats affect performance, or adjust maximum sequence lengths based on your computational resources. With our configuration-driven approach, these experiments become as simple as editing a JSON file rather than diving into code.

### The Magic of HuggingFace Integration

One of the beautiful aspects of modern machine learning is how much infrastructure has been built to make our lives easier. HuggingFace has created an ecosystem where thousands of datasets are just a function call away. But more than convenience, this integration represents a shift toward reproducible, shareable research.

When we call `load_dataset("NebulaSense/Legal_Clause_Instructions")`, we're not just downloading files - we're tapping into a curated, version-controlled dataset that others can use to reproduce our results. This is the kind of scientific rigor that makes machine learning research more reliable and collaborative.

```python
from datasets import load_dataset

def prepare_dataset(
    dataset_name: str,
    instruction_column: str,
    input_column: str, 
    output_column: str,
    sample_size: Optional[int] = None,
    validation_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load and prepare dataset with automatic train/val/test splitting."""
    
    # Load dataset from HuggingFace Hub
    dataset = load_dataset(dataset_name)
    
    # Optional sampling for faster experimentation
    if sample_size is not None:
        dataset["train"] = dataset["train"].select(range(sample_size))
    
    # Automatic train/validation/test splitting
    if validation_size is not None and test_size is not None:
        val_plus_test_size = validation_size + test_size
        split = dataset["train"].train_test_split(test_size=val_plus_test_size, seed=42)
        # ... splitting logic continues
```

Notice how our function handles the practical realities of machine learning experimentation. The `sample_size` parameter acknowledges that you often want to test your pipeline on a smaller subset before committing to a full training run. The automatic train/validation/test splitting ensures that you always have proper evaluation sets, even when working with datasets that don't come pre-split.

### The Language of Structure: Teaching Through Format

Here's where we encounter one of the most elegant insights in supervised fine-tuning: structure is language. The way we format our training examples isn't just about organization - it's about teaching the model a new vocabulary for understanding conversations.

Think about how you might teach someone to follow a recipe. You wouldn't just list ingredients and steps in a random order. Instead, you'd use clear section headers: "Ingredients," "Preparation," "Instructions." These headers do more than organize information - they create a mental framework that helps the reader understand what type of information to expect in each section.

Our instruction format works the same way. When we use headers like `### Instruction`, `### Input`, and `### Output`, we're teaching the model to recognize different types of content and understand their relationships.

```python
def format_instruction_data(data_point: Dict) -> str:
    """Format data into instruction format for fine-tuning."""
    instruction = data_point[instruction_column]
    input_text = data_point[input_column] 
    output = data_point[output_column]
    
    formatted_text = f"### Instruction\n{instruction}\n\n"
    
    if input_text:
        formatted_text += f"### Input\n{input_text}\n\n"
    
    formatted_text += f"### Output\n{output}"
    
    return {"text": formatted_text}

# Apply formatting to entire dataset
train_dataset = dataset["train"].map(
    format_instruction_data, 
    desc="Formatting train data"
)
```

This consistent pattern - `### Instruction` → `### Input` (when relevant) → `### Output` - becomes the model's roadmap for understanding how conversations should flow. After seeing thousands of examples in this format, the model internalize this structure so deeply that it can generate responses that follow the same pattern even for entirely new instructions.

### The Psychology of Consistent Formatting

You might wonder why we're so obsessive about formatting consistency. After all, humans can understand instructions presented in many different ways. We can adapt to various conversation styles, parse meaning from messy text, and infer context from incomplete information. Why can't our models do the same?

The answer lies in how neural networks learn. Unlike humans, who bring years of world experience and contextual understanding to every conversation, our models start with only statistical patterns learned from text. They're incredibly good at recognizing patterns, but they need those patterns to be consistent and clear.

When we use the same section headers across thousands of training examples, we're not just organizing information - we're creating a reliable signal that the model can latch onto. The `### Instruction` header becomes a trigger that tells the model "pay attention, here comes the task you need to understand." The `### Output` header signals "this is the type of response you should generate."

This formatting consistency serves another crucial purpose: it focuses the model's learning. Without clear boundaries between instructions and responses, a model might learn to generate instructions instead of responses, or it might produce rambling outputs that include both the original question and the answer. Our structured format eliminates this ambiguity, creating a clear learning target that produces more reliable, focused responses.

------
## Step 2: The Art of Efficient Model Adaptation

Now we arrive at one of the most ingenious innovations in modern fine-tuning: Parameter-Efficient Fine-Tuning, or PEFT. To understand why this matters, imagine you're a music teacher working with a student who already plays piano beautifully. Instead of making them relearn every song from scratch, you'd focus on teaching them just the specific techniques they need for a new musical style - perhaps some jazz chord progressions or classical fingering patterns.

This is exactly what LoRA (Low-Rank Adaptation) does for language models. Instead of updating all billions of parameters in a large model, we add small "adapter" layers that learn the specific behaviors we want while keeping the original model frozen. It's like teaching new skills without forgetting old ones.

Our implementation takes this concept and wraps it in a configuration-driven framework that makes experimentation effortless. The beauty of this approach is that changing from LoRA to QLoRA (quantized LoRA), switching models, or adjusting training parameters becomes as simple as editing a configuration file.

```python
# config.json
{
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "use_qlora": true,
    "quantization_config": {
        "load_in_4bit": true
    },
    "lora_config": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none"
    }
}
```

This configuration tells a story of careful optimization. The `use_qlora: true` setting indicates we're prioritizing memory efficiency over raw speed, perfect for training on consumer GPUs. The `r: 8` parameter controls the "rank" of our adaptation - essentially how much capacity we're giving our model to learn new behaviors. The `target_modules` list specifies exactly which parts of the transformer we want to adapt, focusing on the attention mechanisms where most of the model's reasoning happens.

### The Elegance of Adaptive Loading

The real magic happens in our `get_apply_peft()` function, which embodies a key principle of good software design: hiding complexity behind simple interfaces. This function makes a sophisticated decision tree look effortless - it can load a model in full precision for LoRA, apply 4-bit quantization for QLoRA, or handle any other configuration we throw at it.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def get_apply_peft(
    model_name: str,
    lora_config: LoraConfig,
    qlora_config: Optional[BitsAndBytesConfig] = None,
) -> torch.nn.Module:
    """Load model and apply PEFT (LoRA/QLoRA) based on configuration."""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=qlora_config,  # None for LoRA, BitsAndBytesConfig for QLoRA
        device_map="auto"
    )
    
    return get_peft_model(model, lora_config)

# Configuration-driven setup
bnb_config = None
if use_qlora:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        load_in_8bit=not quantization_config["load_in_4bit"],
    )

lora_config = LoraConfig(
    **lora_config,  # Unpack from config
    task_type="CAUSAL_LM",
)

# Apply PEFT to model
peft_model = get_apply_peft(model_name, lora_config, bnb_config)
```

What's particularly elegant about this approach is how it handles the decision between LoRA and QLoRA. The `quantization_config` parameter can be `None` (for standard LoRA) or a `BitsAndBytesConfig` object (for QLoRA). The model loading function doesn't need to know which approach we're using - it simply applies whatever quantization configuration we provide. This kind of flexible design makes our code more maintainable and easier to extend as new quantization methods emerge.

### The Professional Touch: Environment Variables and Security

One mark of production-ready code is how it handles sensitive information. Rather than hardcoding API tokens or usernames directly into our scripts (a security nightmare), we use environment variables to keep credentials separate from code. This approach isn't just about security - it's about creating code that can work seamlessly across different environments and users.

```python
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME") 
login(HF_TOKEN)

# Model name for saving adapters
adapter_model_name = f"{HF_USERNAME}/{save_model_name}"
```

This pattern exemplifies thoughtful software design. The `.env` file keeps sensitive tokens secure and out of version control, while the automatic login and model naming make sharing fine-tuned adapters effortless. When training completes, our adapters will be automatically uploaded to HuggingFace Hub with a name that clearly identifies both the creator and the model variant.

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

### Integrated Tokenization and Dataset Pipeline

Our implementation combines tokenization with dataset preparation in a single pipeline:

```python
def tokenize_dataset(
    model_name: str,
    dataset_name: str, 
    instruction_column: str,
    input_column: str,
    output_column: str,
    assistant_only_masking: bool = True,
    max_length: int = 2048,
    sample_size: Optional[int] = None,
    validation_size: Optional[float] = None,
    test_size: Optional[float] = None,
) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """Complete pipeline: load → format → tokenize → mask."""
    
    # Step 1: Load and format dataset
    train_dataset, validation_dataset, test_dataset = prepare_dataset(
        dataset_name, instruction_column, input_column, output_column,
        sample_size, validation_size, test_size
    )
    
    # Step 2: Setup tokenizer with proper padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Step 3: Tokenization with masking
    def tokenize_and_mask_function(examples):
        texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
        tokenized = tokenizer(
            texts_with_eos,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True,
        )
        
        # Apply assistant-only masking (covered in next section)
        labels = []
        for input_ids in tokenized["input_ids"]:
            if assistant_only_masking:
                masked_labels = apply_assistant_masking(input_ids, tokenizer)
            else:
                masked_labels = input_ids
            labels.append(masked_labels)
        
        tokenized["labels"] = labels
        return tokenized
    
    # Step 4: Apply to all dataset splits
    train = train_dataset.map(
        tokenize_and_mask_function, batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing train dataset"
    )
    # ... similar for validation and test
    
    return train, validation, test, tokenizer
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

## Step 6: Configuration-Driven Training with Early Stopping

Our implementation uses the standard Transformers `Trainer` with custom data collation and optional early stopping, all controlled through configuration.

### Configuration-Based Training Setup

All training parameters are defined in `config.json`:

```python
# config.json
{
    "training_args": {
        "output_dir": "./checkpoints",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "logging_steps": 4,
        "save_strategy": "steps",
        "save_steps": 50,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "max_grad_norm": 1.0,
        "load_best_model_at_end": true,
        "metric_for_best_model": "eval_loss"
    },
    "early_stopping": {
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.01
    }
}
```

### Custom Data Collator for Efficient Padding

Instead of using a generic data collator, we implement custom padding logic:

```python
class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Remove labels before padding
        labels = [f.pop("labels") for f in features]
        
        # Pad input_ids and attention_mask consistently
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        
        # Pad labels to same max length with -100 (ignored in loss)
        max_len = batch["input_ids"].size(1)
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            padded_labels[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        batch["labels"] = padded_labels
        return batch

data_collator = DataCollatorForCausalLM(tokenizer)
```

### Training with Early Stopping

The training setup integrates early stopping and configuration management:

```python
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# Load all configuration
config = read_json_file(CONFIG_FILE)
training_args = config["training_args"]
early_stopping_config = config.get("early_stopping", {})

# Setup training arguments
training_args = TrainingArguments(**training_args)

# Setup early stopping callback if configured
callbacks = []
if early_stopping_config:
    callbacks.append(EarlyStoppingCallback(**early_stopping_config))

# Initialize trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train,
    eval_dataset=validation,
    data_collator=data_collator,
    callbacks=callbacks,
)

# Start training
trainer.train()
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

### Automatic Model Saving and Sharing

Our implementation automatically saves adapters locally and pushes them to HuggingFace Hub:

```python
# After training completes, save and push adapters
print("SAVING AND PUSHING ADAPTERS")

# Generate model name from config
adapter_model_name = f"{HF_USERNAME}/{save_model_name}"

# Save adapters locally first
local_adapter_path = "final_adapters"
peft_model.save_pretrained(local_adapter_path)
tokenizer.save_pretrained(local_adapter_path)

print(f"Adapters saved locally to: {local_adapter_path}")

# Push to Hugging Face Hub with error handling
try:
    peft_model.push_to_hub(adapter_model_name, private=False)
    tokenizer.push_to_hub(adapter_model_name)
    print(f"✅ Adapters successfully pushed to: https://huggingface.co/{adapter_model_name}")
except Exception as e:
    print(f"❌ Error pushing to Hugging Face: {e}")
    print("Make sure you're logged in with: huggingface-cli login")
```

### Complete Configuration Example

Here's a complete `config.json` that brings everything together:

```json
{
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "save_model_name": "llama-1b-legal-qlora",
    "assistant_only_masking": false,
    "use_qlora": true,
    "dataset_config": {
        "dataset_name": "NebulaSense/Legal_Clause_Instructions",
        "instruction_column": "Instruction",
        "input_column": "Input", 
        "output_column": "Output",
        "max_length": 2048
    },
    "quantization_config": {
        "load_in_4bit": true
    },
    "lora_config": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none"
    },
    "training_args": {
        "output_dir": "./checkpoints",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "learning_rate": 2e-4,
        "logging_steps": 4,
        "save_strategy": "steps",
        "save_steps": 50,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "max_grad_norm": 1.0,
        "load_best_model_at_end": true,
        "metric_for_best_model": "eval_loss"
    },
    "early_stopping": {
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.01
    }
}
```

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

- **Configuration-driven development** - using `config.json` makes experiments reproducible and easy to modify without changing code
- **Integrated pipeline approach** - our `tokenize_dataset()` function combines loading, formatting, tokenization, and masking in one step
- **Flexible LoRA/QLoRA setup** - the `get_apply_peft()` function handles both approaches based on configuration
- **Custom data collation** - implementing `DataCollatorForCausalLM` gives precise control over padding and label handling
- **Early stopping prevents overfitting** - monitoring validation loss and stopping when improvement plateaus saves time and improves generalization
- **Automatic model sharing** - built-in HuggingFace Hub integration makes sharing fine-tuned adapters seamless
- **Environment variable integration** - using `.env` files keeps sensitive tokens secure while enabling automated workflows

This production-ready implementation gives you a complete fine-tuning pipeline that's both flexible and robust. The configuration-driven approach means you can experiment with different models, datasets, and training parameters by simply editing JSON files.

In our next lesson, we'll explore advanced optimization techniques and evaluation methods that build on this solid foundation.

---

[⬅️ Previous - TBD](TBD)
[➡️ Next - TBD](TBD)

---
