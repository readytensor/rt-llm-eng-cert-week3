import os
import argparse
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator
)

class LossTrackingCallback(TrainerCallback):
    """Custom callback to track training and validation losses during training."""
    
    def __init__(self, patience=2, threshold=0.01):
        self.training_losses = []
        self.eval_losses = []
        self.eval_steps = []  # Track the step number for each evaluation
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience
        self.threshold = threshold
        self.early_stopping_triggered = False
        self.current_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture losses from logs."""
        logs = logs or {}
        
        # Track training loss
        if "loss" in logs:
            self.training_losses.append(logs["loss"])
            self.current_step += 1
            
        # Track evaluation loss
        if "eval_loss" in logs:
            current_loss = logs["eval_loss"]
            self.eval_losses.append(current_loss)
            self.eval_steps.append(self.current_step)
            
            # Early stopping logic
            if current_loss < (self.best_loss - self.threshold):
                # Loss is improving by more than the threshold
                self.best_loss = current_loss
                self.counter = 0
                print(f"\nValidation loss improved to {current_loss:.4f}")
            else:
                # Loss is not improving by more than the threshold
                self.counter += 1
                print(f"\nValidation loss did not improve significantly: {current_loss:.4f}. Counter: {self.counter}/{self.patience}")
                
                if self.counter >= self.patience and not self.early_stopping_triggered:
                    self.early_stopping_triggered = True
                    print(f"\n*** Early stopping triggered after {self.patience} evaluations without significant improvement ***")
                    print(f"Best validation loss: {self.best_loss:.4f}, Current validation loss: {current_loss:.4f}")
                    control.should_training_stop = True
    
    def plot_losses(self, output_dir):
        """Generate a plot of training and validation losses."""
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.training_losses:
            plt.plot(range(len(self.training_losses)), self.training_losses, label='Training Loss')
        
        # Plot validation loss
        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, 'o-', label='Validation Loss')
            
            # Find where early stopping would occur (minimum validation loss)
            min_loss_idx = self.eval_losses.index(min(self.eval_losses))
            min_loss_step = self.eval_steps[min_loss_idx]
            plt.axvline(x=min_loss_step, color='r', linestyle='--', 
                        label=f'Early Stopping Point (Step {min_loss_step})')
            
            # Add text to explain early stopping
            if self.early_stopping_triggered:
                plt.text(min_loss_step + 5, min(self.eval_losses), 
                         "Early stopping triggered\nValidation loss stopped improving",
                         bbox=dict(facecolor='red', alpha=0.2))
            
            # Add overfitting indicator if validation loss increases
            if len(self.eval_losses) >= 2:
                for i in range(1, len(self.eval_losses)):
                    if self.eval_losses[i] > self.eval_losses[i-1]:
                        plt.text(self.eval_steps[i], self.eval_losses[i], 
                                "Possible overfitting",
                                bbox=dict(facecolor='yellow', alpha=0.2))
                        break
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss with Early Stopping')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'early_stopping_plot.png'))
        plt.close()
        
        print(f"Loss plot saved to {os.path.join(output_dir, 'early_stopping_plot.png')}")


def tokenize_function(examples, tokenizer, max_length=128):
    """Tokenize the dataset examples with proper labels for causal language modeling."""
    # Tokenize inputs
    result = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    
    # Create labels identical to input_ids for causal language modeling
    result["labels"] = result["input_ids"].copy()
    
    return result


def run_training_with_early_stopping(output_dir, model_name="distilgpt2"):
    """Run a full training with early stopping and generate visualizations."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Make sure tokenizer has pad token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset: wikitext-2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize the dataset with labels for causal language modeling
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Create an artificial small dataset to force early stopping
    # Using a very small dataset will cause the model to overfit quickly
    train_dataset = tokenized_datasets["train"].select(range(200))
    
    # Create a validation set that's slightly different from training
    # This will cause validation loss to plateau or increase while training loss keeps decreasing
    eval_dataset = tokenized_datasets["validation"].select(range(50))
    
    # Create custom loss tracking callback
    loss_tracker = LossTrackingCallback(patience=2, threshold=0.01)
    
    # Configure training arguments with settings to trigger early stopping
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,  # Set high, early stopping will kick in
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        # Evaluate frequently to trigger early stopping
        eval_steps=20,
        eval_strategy="steps",
        save_steps=20,
        save_strategy="steps",
        save_total_limit=2,
        learning_rate=5e-4,  # Higher learning rate to reach plateau faster
        # Required for early stopping to work
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Define a custom data collator that handles the padding correctly
    def data_collator(features):
        batch = default_data_collator(features)
        
        # Move the padding token ID to -100 in labels, so it's ignored in loss calculation
        if "labels" in batch:
            labels = batch["labels"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        return batch
    
    # Create trainer with our custom callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[loss_tracker]
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Generate and save the loss plot
    loss_tracker.plot_losses(output_dir)
    
    # Save the best model
    trainer.save_model(f"{output_dir}/best_model")
    print(f"Best model saved to {output_dir}/best_model")
    
    return loss_tracker


def run_simulated_visualization(output_dir):
    """Create a simulated visualization of early stopping without actual training."""
    # Create loss values that clearly demonstrate early stopping
    # Training loss continuously decreases
    train_steps = 100
    train_losses = [4.5 - 0.03*i for i in range(train_steps)]
    
    # Validation loss decreases, plateaus, then increases
    eval_steps = [20, 40, 60, 80, 100]
    val_losses = [4.6, 4.4, 4.5, 4.7, 5.0]  # Loss plateaus then increases
    
    # Find where early stopping would occur (minimum validation loss)
    min_loss_idx = val_losses.index(min(val_losses))
    min_loss_step = eval_steps[min_loss_idx]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(range(train_steps), train_losses, label='Training Loss')
    plt.plot(eval_steps, val_losses, 'o-', label='Validation Loss')
    
    # Add vertical line for early stopping point
    plt.axvline(x=min_loss_step, color='r', linestyle='--', 
                label=f'Early Stopping Point (Step {min_loss_step})')
    
    # Add annotation for early stopping
    plt.text(min_loss_step + 5, val_losses[min_loss_idx], 
             "Early stopping\nBest validation loss", 
             bbox=dict(facecolor='red', alpha=0.2))
    
    # Add annotation for overfitting
    plt.text(eval_steps[3], val_losses[3], 
             "Overfitting begins\nValidation loss increases", 
             bbox=dict(facecolor='yellow', alpha=0.2))
    
    # Add title and labels
    plt.title('Simulated Training Process with Early Stopping')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'simulated_early_stopping.png'))
    plt.close()
    
    print(f"Simulated early stopping visualization saved to {os.path.join(output_dir, 'simulated_early_stopping.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or visualize early stopping")
    parser.add_argument("--output-dir", type=str, default="./outputs/early_stopping_demo",
                        help="Directory to save outputs")
    parser.add_argument("--demo-type", type=str, choices=["full", "visualization"], default="full",
                        help="Type of demo to run: 'full' for actual training, 'visualization' for simulated plot")
    parser.add_argument("--model-name", type=str, default="distilgpt2",
                        help="Model to use for training (default: distilgpt2)")
    
    args = parser.parse_args()
    
    if args.demo_type == "full":
        # Run actual training with early stopping
        loss_tracker = run_training_with_early_stopping(
            output_dir=args.output_dir,
            model_name=args.model_name
        )
    else:
        # Create simulated visualization
        run_simulated_visualization(args.output_dir)