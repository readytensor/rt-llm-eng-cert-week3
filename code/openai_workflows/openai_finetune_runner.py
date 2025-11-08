"""
openai_finetune_runner.py
Create and monitor fine-tuning jobs for OpenAI models.
Used for Lesson 2 (Fine-Tuning Frontier LLMs).
"""

import os
import time
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

from utils.config_utils import load_config
from paths import EXPERIMENTS_DIR, OPENAI_FILES_DIR


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
client = OpenAI()

OPENAI_TRAIN_DIR = os.path.join(EXPERIMENTS_DIR, "openai_finetune")
os.makedirs(OPENAI_TRAIN_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def upload_file(file_path):
    """Upload a JSONL file for fine-tuning."""
    print(f"📤 Uploading file: {file_path}")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    print(f"✅ Uploaded file ID: {response.id}")
    return response.id


def create_finetune_job(
    base_model, training_file_id, validation_file_id=None, suffix="samsum-ft"
):
    """Create a new fine-tuning job."""
    print(f"\n🚀 Creating fine-tuning job for {base_model}...")

    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=base_model,
        suffix=suffix,
    )

    print(f"✅ Fine-tune job created: {job.id}")
    return job


def monitor_finetune_job(job_id, refresh_interval=30):
    """Monitor job progress until completion."""
    print(f"📊 Monitoring fine-tune job: {job_id}")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"🕒 Status: {status} | Trained tokens: {job.trained_tokens or 0}")

        if status in ("succeeded", "failed", "cancelled"):
            print(
                f"\n🏁 Job {status.upper()} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if status == "succeeded":
                print(f"✅ Fine-tuned model ID: {job.fine_tuned_model}")
            break

        time.sleep(refresh_interval)

    return job


def list_finetune_jobs(limit=10):
    """List recent fine-tuning jobs."""
    print(f"\n📜 Listing the last {limit} fine-tuning jobs:")
    jobs = client.fine_tuning.jobs.list(limit=limit)
    for j in jobs.data:
        print(f"- {j.id} | {j.model} → {j.fine_tuned_model or 'N/A'} | {j.status}")
    return jobs


def list_finetuned_models():
    """List available fine-tuned models."""
    print("\n🤖 Listing fine-tuned models:")
    models = client.models.list()
    for m in models.data:
        if "ft:" in m.id:
            print(f"- {m.id}")
    return models


def save_job_metadata(job, tag="latest"):
    """Save job metadata locally."""
    job_info = job.model_dump() if hasattr(job, "model_dump") else job
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(OPENAI_TRAIN_DIR, f"{tag}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2)
    print(f"💾 Job metadata saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Manage OpenAI fine-tuning jobs.")
    parser.add_argument("--train_file", type=str, help="Path to training JSONL file")
    parser.add_argument(
        "--val_file", type=str, default=None, help="Optional validation JSONL file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="samsum-ft",
        help="Custom name suffix for fine-tuned model",
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Monitor the created fine-tuning job"
    )
    parser.add_argument(
        "--list_jobs", action="store_true", help="List recent fine-tuning jobs"
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List fine-tuned models"
    )
    args = parser.parse_args()

    cfg = load_config()

    # Handle listing commands
    if args.list_jobs:
        list_finetune_jobs()
        return
    if args.list_models:
        list_finetuned_models()
        return

    # ✅ Default behavior if no args provided
    if not args.train_file and not args.list_jobs and not args.list_models:
        print("⚙️  No arguments provided — using defaults from config.")
        dataset_name = cfg["datasets"][0]["path"].replace("/", "_")
        base_path = os.path.join(OPENAI_FILES_DIR, dataset_name)

        args.train_file = os.path.join(base_path, "train.jsonl")
        args.val_file = os.path.join(base_path, "validation.jsonl")
        args.base_model = cfg.get("openai_base_model", "gpt-4o-mini-2024-07-18")
        args.suffix = "samsum-ft"
        args.monitor = True

    # Upload files
    train_file_id = upload_file(args.train_file)
    val_file_id = (
        upload_file(args.val_file)
        if args.val_file and os.path.exists(args.val_file)
        else None
    )

    # Create job
    job = create_finetune_job(
        base_model=args.base_model,
        training_file_id=train_file_id,
        validation_file_id=val_file_id,
        suffix=args.suffix,
    )

    save_job_metadata(job, tag="created_job")

    if args.monitor:
        job = monitor_finetune_job(job.id)
        save_job_metadata(job, tag="final_status")


if __name__ == "__main__":
    main()
