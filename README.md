# Week 3: Fine-Tuning LLMs

**LLM Engineering and Development Certification Program**

This repository contains code and materials for Week 3, where we learn to fine-tune large language models and measure improvements over baseline performance.

---

## 📚 Week 3 Overview

**Goal**: Take a base model, fine-tune it using different approaches, and measure improvement.

### Lessons Covered

- **Lesson 1**: Dataset Selection & Baseline Evaluation
- **Lesson 2**: Fine-Tuning Frontier LLMs (OpenAI)
- **Lesson 3**: End-to-End LoRA Fine-Tuning
- **Lesson 4**: Experiment Tracking & Reproducibility (W&B) _(Grid search - in progress)_
- **Lessons 5-8**: Advanced topics _(coming soon)_

---

## 🏗️ Repository Structure

```
.
├── code/
│   ├── config.yaml                    # Main configuration file
│   ├── paths.py                       # Centralized path management
│   │
│   ├── evaluate_baseline.py           # Lesson 1: Baseline evaluation
│   ├── train_lora.py                  # Lesson 3: LoRA fine-tuning
│   ├── evaluate_lora.py               # Lesson 3: Evaluate fine-tuned model
│   │
│   ├── openai_workflow.py             # Lesson 2: OpenAI workflow controller
│   ├── openai_workflows/              # Lesson 2: OpenAI fine-tuning scripts
│   │   ├── prepare_openai_jsonl.py
│   │   ├── openai_finetune_runner.py
│   │   └── evaluate_openai.py
│   │
│   ├── run_grid_search.py             # Lesson 4: Grid search (WIP)
│   │
│   └── utils/                         # Shared utilities
│       ├── config_utils.py            # Config loading
│       ├── data_utils.py              # Dataset loading & preprocessing
│       ├── model_utils.py             # Model setup & management
│       └── inference_utils.py         # Generation & evaluation
│
├── data/
│   ├── datasets/                      # Cached HuggingFace datasets
│   ├── outputs/                       # All evaluation results
│   │   ├── baseline/                  # Lesson 1 results
│   │   ├── lora_samsum/              # Lesson 3 results
│   │   └── openai/                   # Lesson 2 results
│   └── experiments/                   # OpenAI fine-tuning artifacts
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## ⚙️ Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
# For OpenAI fine-tuning (Lesson 2)
OPENAI_API_KEY=your_openai_api_key_here

# For Weights & Biases tracking (Lesson 4)
WANDB_API_KEY=your_wandb_api_key_here

# Optional: For Hugging Face model uploads
HF_TOKEN=your_huggingface_token_here
```

### 4. Review Configuration

Edit `code/config.yaml` to customize:

- Base model (default: `meta-llama/Llama-3.2-1B-Instruct`)
- Dataset (default: `knkarthick/samsum`)
- Training hyperparameters
- LoRA configuration

---

## 🚀 Usage

### **Lesson 1: Baseline Evaluation**

Evaluate the base model (no fine-tuning) to establish baseline performance.

```bash
cd code
python evaluate_baseline.py
```

**Output**:

- Results saved to `data/outputs/baseline/eval_results.json`
- Predictions saved to `data/outputs/baseline/predictions.jsonl`

**Expected ROUGE-1**: ~34% (on SAMSum dataset)

---

### **Lesson 2: Fine-Tuning Frontier LLMs (OpenAI)**

Complete workflow for fine-tuning OpenAI models like GPT-4o-mini.

#### Interactive Workflow

```bash
cd code
python openai_workflow.py
```

This launches an interactive menu:

1. Prepare dataset for fine-tuning
2. Run fine-tuning job
3. Evaluate base or fine-tuned model
4. Exit

#### Or Run Individual Steps

**Step 1: Prepare Data**

```bash
python openai_workflows/prepare_openai_jsonl.py
```

**Step 2: Create Fine-Tuning Job**

```bash
python openai_workflows/openai_finetune_runner.py
```

This will:

- Upload training/validation files
- Create fine-tuning job
- Monitor progress until completion
- Save fine-tuned model ID

**Step 3: Evaluate Base Model**

```bash
python openai_workflows/evaluate_openai.py --model gpt-4o-mini
```

**Step 4: Evaluate Fine-Tuned Model**

```bash
python openai_workflows/evaluate_openai.py --model ft:gpt-4o-mini-2024-07-18:your-org:model-name:job-id
```

**Output**:

- Results saved to `data/outputs/openai/{model_name}/`

---

### **Lesson 3: End-to-End LoRA Fine-Tuning**

Fine-tune Llama using QLoRA (4-bit quantization + LoRA adapters).

#### Step 1: Train Model

```bash
cd code
python train_lora.py
```

**What happens**:

- Loads base model with 4-bit quantization
- Applies LoRA adapters to attention layers
- Fine-tunes on SAMSum dataset
- Logs metrics to Weights & Biases
- Saves adapters to `data/outputs/lora_samsum/lora_adapters/`

**Training time**: ~15-20 minutes on a single GPU (RTX 3090 / A100)

#### Step 2: Evaluate Fine-Tuned Model

```bash
python evaluate_lora.py
```

**Output**:

- Results saved to `data/outputs/lora_samsum/eval_results.json`
- Predictions saved to `data/outputs/lora_samsum/predictions.jsonl`

**Expected improvement**: ROUGE-1 should increase by ~5-10% over baseline

---

## 🔧 Configuration

All configuration is centralized in `code/config.yaml`:

### Change the Base Model

```yaml
base_model: meta-llama/Llama-3.2-3B-Instruct # or any HF model
```

### Change the Dataset

```yaml
datasets:
  - path: your-org/your-dataset
    cache_dir: ../data/datasets
    field_map:
      input: dialogue # Your input field name
      output: summary # Your output field name
    type: completion
```

### Adjust Training Hyperparameters

```yaml
num_epochs: 3
learning_rate: 2e-4
batch_size: 4
gradient_accumulation_steps: 4
```

### Modify LoRA Configuration

```yaml
lora_r: 8 # Rank (higher = more parameters)
lora_alpha: 16 # Scaling factor
lora_dropout: 0.1
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

---

## 📊 Results Comparison

After completing lessons 1-3, compare results:

| Model                                 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------------------------------- | ------- | ------- | ------- |
| **Baseline** (Lesson 1)               | ~34%    | ~12%    | ~27%    |
| **OpenAI GPT-4o-mini** (Lesson 2)     | ~41%    | ~16%    | ~32%    |
| **Fine-tuned GPT-4o-mini** (Lesson 2) | ~53%    | ~28%    | ~45%    |
| **Fine-tuned Llama LoRA** (Lesson 3)  | TBD     | TBD     | TBD     |

_Run each lesson to populate your own results!_

---

## 🧪 Lesson 4: Grid Search (Work in Progress)

```bash
# Note: This script is not yet verified
python run_grid_search.py
```

This will:

- Systematically test different LoRA hyperparameters
- Log all experiments to Weights & Biases
- Save results for comparison

---

## 🤝 Contributing

This is an educational repository. Feel free to:

- Open issues for bugs or questions
- Submit PRs for improvements
- Share your fine-tuning results!

---

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning) Guide](https://huggingface.co/docs/peft)
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

---

## 📄 License

This project is licensed under the CC BY-NC-SA 4.0 License - see the [LICENSE](LICENSE) file for details.

## Contact

**Ready Tensor, Inc.**

- Email: contact at readytensor dot com
- Issues & Contributions: Open an issue or pull request on this repository
- Website: [Ready Tensor](https://readytensor.ai)
