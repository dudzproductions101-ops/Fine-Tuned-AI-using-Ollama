"""
training/train.py
-----------------
Fine-tunes CodeLlama using QLoRA (Quantized Low-Rank Adaptation).
Supports 4-bit quantization for running on consumer GPUs (16GB VRAM).
Falls back gracefully to CPU (slow) if no GPU available.

Training Stack:
  - transformers  : Model loading + Trainer
  - peft          : LoRA adapter injection
  - bitsandbytes  : 4-bit quantization
  - trl           : SFTTrainer (handles packing, padding, etc.)
  - accelerate    : Multi-GPU / mixed precision support

Usage:
    python training/train.py --config configs/config.yaml
    python training/train.py --config configs/config.yaml --resume outputs/fine_tuned
    python training/train.py --dataset data/final --output outputs/fine_tuned --epochs 3
"""

import sys
import os
import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

# Transformers + PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, Dataset, DatasetDict
import datasets as hf_datasets

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config

log = get_logger("trainer")
fm = FileManager()


# ============================================================
# Hardware Detection
# ============================================================

def get_device_info() -> Dict:
    """Detect available hardware and return optimal settings."""
    info = {
        "has_cuda": torch.cuda.is_available(),
        "has_mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "recommended_batch_size": 1,
    }

    if info["has_cuda"]:
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram = info["gpu_memory_gb"]

        if vram >= 40:
            info["recommended_batch_size"] = 8
        elif vram >= 24:
            info["recommended_batch_size"] = 4
        elif vram >= 16:
            info["recommended_batch_size"] = 2
        else:
            info["recommended_batch_size"] = 1

    elif info["has_mps"]:
        info["device"] = "mps"
        info["recommended_batch_size"] = 1

    return info


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer(config: Dict, device_info: Dict):
    """
    Load the base model with optional 4-bit quantization.
    Injects LoRA adapters for parameter-efficient fine-tuning.

    Returns:
        (model, tokenizer)
    """
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    base_model_name = model_cfg["base_model"]

    log.info(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_fast=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    log.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Set up quantization config
    compute_dtype = torch.float16

    bnb_config = None
    if quant_cfg.get("use_4bit") and device_info["has_cuda"]:
        log.info("Setting up 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("use_nested_quant", True),
        )
    else:
        log.info("Loading model in full precision (no GPU quantization).")

    log.info(f"Loading model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto" if device_info["has_cuda"] else None,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=compute_dtype if device_info["has_cuda"] else torch.float32,
    )

    # Prepare model for k-bit training (required for 4-bit QLoRA)
    if bnb_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True)
        )

    # Inject LoRA adapters
    log.info(f"Injecting LoRA adapters (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})...")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# Dataset Loading
# ============================================================

def load_training_dataset(dataset_path: str, tokenizer, max_length: int = 2048) -> DatasetDict:
    """
    Load the prepared dataset.
    Tries HuggingFace Dataset format first, falls back to JSONL.

    Args:
        dataset_path: Path to data/final directory
        tokenizer: Tokenizer (for validation)
        max_length: Maximum sequence length

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    dataset_path = Path(dataset_path)
    hf_path = dataset_path / "hf_dataset"

    if hf_path.exists():
        log.info(f"Loading HuggingFace Dataset from {hf_path}")
        dataset = load_from_disk(str(hf_path))
    else:
        # Load from JSONL
        log.info(f"Loading from JSONL files in {dataset_path}")
        train_file = dataset_path / "train.jsonl"
        val_file = dataset_path / "validation.jsonl"

        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")

        train_data = fm.read_jsonl(train_file)
        val_data = fm.read_jsonl(val_file) if val_file.exists() else train_data[:100]

        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
        })

    log.info(f"Dataset loaded:")
    log.info(f"  Train: {len(dataset['train'])} examples")
    if "validation" in dataset:
        log.info(f"  Validation: {len(dataset['validation'])} examples")

    return dataset


# ============================================================
# Training Arguments Builder
# ============================================================

def build_training_args(config: Dict, output_dir: str, device_info: Dict) -> TrainingArguments:
    """Build TrainingArguments from config."""
    train_cfg = config["training"]

    batch_size = train_cfg.get("per_device_train_batch_size", device_info["recommended_batch_size"])

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_32bit"),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        fp16=train_cfg.get("fp16", False) and device_info["has_cuda"],
        bf16=train_cfg.get("bf16", False) and device_info["has_cuda"],
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        max_steps=train_cfg.get("max_steps", -1),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        group_by_length=train_cfg.get("group_by_length", True),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 25),
        evaluation_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=2,
        dataloader_pin_memory=device_info["has_cuda"],
    )


# ============================================================
# Main Training Function
# ============================================================

def train(config: Dict, dataset_path: str, output_dir: str, resume_from: Optional[str] = None):
    """
    Main training entrypoint.

    Args:
        config: Loaded YAML config dict
        dataset_path: Path to prepared dataset (data/final)
        output_dir: Where to save checkpoints and final model
        resume_from: Optional checkpoint path to resume from
    """
    # Detect hardware
    device_info = get_device_info()
    log.info("Hardware Info:")
    for k, v in device_info.items():
        log.info(f"  {k}: {v}")

    fm._ensure_dir(output_dir)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device_info)
    max_length = config["model"].get("max_length", 2048)

    # Load dataset
    dataset = load_training_dataset(dataset_path, tokenizer, max_length)

    # Build training arguments
    training_args = build_training_args(config, output_dir, device_info)

    # The SFTTrainer handles:
    # - packing (multiple short examples into one sequence)
    # - data collation
    # - training loop with all callbacks
    log.info("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=max_length,
        packing=True,           # Pack multiple short examples into one sequence
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    # Resume from checkpoint if specified
    checkpoint = resume_from if resume_from and Path(resume_from).exists() else None
    if checkpoint:
        log.info(f"Resuming from checkpoint: {checkpoint}")
    else:
        log.info("Starting training from scratch...")

    # Train
    log.info(f"Training started. Output: {output_dir}")
    log.info(f"Epochs: {training_args.num_train_epochs}")
    log.info(f"Steps per epoch: {math.ceil(len(dataset['train']) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    log.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    fm.write_json(Path(output_dir) / "training_metrics.json", metrics)

    log.info(f"\n{'='*50}")
    log.info("TRAINING COMPLETE")
    log.info(f"Model saved to: {output_dir}")
    log.info(f"Final loss: {metrics.get('train_loss', 'N/A'):.4f}")

    return metrics


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CodeLlama with QLoRA")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--dataset", default=None, help="Override dataset path (data/final)")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.dataset:
        config["dataset"]["data_dir"] = args.dataset
    if args.output:
        config["training"]["output_dir"] = args.output
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size

    dataset_path = args.dataset or "data/final"
    output_dir = args.output or config["training"]["output_dir"]

    metrics = train(
        config=config,
        dataset_path=dataset_path,
        output_dir=output_dir,
        resume_from=args.resume
    )

    log.info(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
