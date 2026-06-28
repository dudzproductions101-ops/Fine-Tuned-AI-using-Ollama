"""
dataset/converter.py
--------------------
Converts training pairs JSONL to HuggingFace Dataset format
with train/validation/test splits.

Also:
  - Applies the prompt template from config
  - Tokenizes and checks length distribution
  - Saves as HF Dataset (Arrow format) and optionally as JSONL splits

Usage:
    python dataset/converter.py --input data/dataset --output data/final --format both
    python dataset/converter.py --input data/dataset --output data/final --format jsonl
    python dataset/converter.py --input data/dataset --output data/final --format hf
"""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config

log = get_logger("converter")
fm = FileManager()


# ============================================================
# Prompt Formatter
# ============================================================

class PromptFormatter:
    """
    Formats instruction/input/output into the model's expected prompt format.
    Supports CodeLlama Instruct format and Alpaca format.
    """

    # CodeLlama Instruct format
    CODELLAMA_TEMPLATE = "[INST] {instruction}{input_section} [/INST]\n{output}"

    # Alpaca format (used by many open models)
    ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}
{input_section}
### Response:
{output}"""

    # Chat format
    CHAT_TEMPLATE = "<|user|>\n{instruction}{input_section}\n<|assistant|>\n{output}"

    TEMPLATES = {
        "codellama": CODELLAMA_TEMPLATE,
        "alpaca": ALPACA_TEMPLATE,
        "chat": CHAT_TEMPLATE,
    }

    def __init__(self, template_name: str = "codellama"):
        self.template_name = template_name
        self.template = self.TEMPLATES.get(template_name, self.CODELLAMA_TEMPLATE)

    def format(self, example: Dict) -> str:
        """Format an instruction/input/output example into the full prompt."""
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        output = example.get("output", "").strip()

        input_section = f"\n\n### Input:\n{input_text}" if input_text else ""

        return self.template.format(
            instruction=instruction,
            input_section=input_section,
            output=output
        )

    def format_inference(self, instruction: str, input_text: str = "") -> str:
        """Format for inference (no output)."""
        input_section = f"\n\n### Input:\n{input_text}" if input_text else ""

        if self.template_name == "codellama":
            return f"[INST] {instruction}{input_section} [/INST]"
        elif self.template_name == "alpaca":
            return f"""Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}
{input_section}
### Response:"""
        elif self.template_name == "chat":
            return f"<|user|>\n{instruction}{input_section}\n<|assistant|>"
        return f"{instruction}{input_section}"


# ============================================================
# Length Analysis
# ============================================================

def analyze_lengths(examples: List[Dict], formatter: PromptFormatter) -> Dict:
    """Analyze token/character length distribution of the dataset."""
    lengths = []
    for ex in tqdm(examples, desc="Analyzing lengths", leave=False):
        text = formatter.format(ex)
        # Approx 4 chars per token
        lengths.append(len(text) // 4)

    lengths.sort()
    n = len(lengths)

    stats = {
        "count": n,
        "min": lengths[0] if n else 0,
        "max": lengths[-1] if n else 0,
        "mean": sum(lengths) / n if n else 0,
        "p50": lengths[n // 2] if n else 0,
        "p90": lengths[int(n * 0.9)] if n else 0,
        "p95": lengths[int(n * 0.95)] if n else 0,
        "p99": lengths[int(n * 0.99)] if n else 0,
    }

    log.info("Length Distribution (approx tokens):")
    for k, v in stats.items():
        log.info(f"  {k}: {v:.0f}")

    return stats


# ============================================================
# Dataset Converter
# ============================================================

class DatasetConverter:
    """
    Reads training pairs JSONL, applies prompt formatting,
    splits into train/val/test, saves as HF Dataset and/or JSONL.
    """

    def __init__(self, config: Dict):
        self.config = config
        dataset_cfg = config.get("dataset", {})

        self.train_split = dataset_cfg.get("train_split", 0.90)
        self.val_split = dataset_cfg.get("val_split", 0.05)
        self.test_split = dataset_cfg.get("test_split", 0.05)
        self.shuffle = dataset_cfg.get("shuffle", True)
        self.seed = dataset_cfg.get("seed", 42)
        self.max_samples = dataset_cfg.get("max_samples", None)

        model_cfg = config.get("model", {})
        self.max_length = model_cfg.get("max_length", 2048)

        # Detect template format from model
        base_model = model_cfg.get("base_model", "codellama")
        if "codellama" in base_model.lower():
            template = "codellama"
        elif "llama" in base_model.lower():
            template = "chat"
        else:
            template = "alpaca"

        self.formatter = PromptFormatter(template)
        log.info(f"Using prompt template: {template}")

    def _load_examples(self, input_dir: Path) -> List[Dict]:
        """Load all examples from JSONL files in input_dir."""
        jsonl_files = list(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files in {input_dir}")

        examples = []
        for f in jsonl_files:
            records = fm.read_jsonl(f)
            examples.extend(records)
            log.info(f"Loaded {len(records)} examples from {f.name}")

        log.info(f"Total examples loaded: {len(examples)}")
        return examples

    def _split(self, examples: List[Dict]) -> Tuple[List, List, List]:
        """Split examples into train/val/test."""
        if self.shuffle:
            random.Random(self.seed).shuffle(examples)

        if self.max_samples and len(examples) > self.max_samples:
            examples = examples[:self.max_samples]
            log.info(f"Capped at {self.max_samples} samples")

        n = len(examples)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        train = examples[:train_end]
        val = examples[train_end:val_end]
        test = examples[val_end:]

        log.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _format_for_training(self, examples: List[Dict]) -> List[Dict]:
        """Apply prompt template to all examples."""
        formatted = []
        skipped = 0

        for ex in tqdm(examples, desc="Formatting", leave=False):
            text = self.formatter.format(ex)
            # Rough token check (4 chars/token)
            if len(text) // 4 > self.max_length:
                skipped += 1
                continue
            formatted.append({
                "text": text,
                "instruction": ex.get("instruction", ""),
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
            })

        if skipped:
            log.warning(f"Skipped {skipped} examples exceeding max_length={self.max_length} tokens")

        return formatted

    def _save_jsonl_splits(self, splits: Dict[str, List], output_dir: Path) -> None:
        """Save train/val/test as JSONL files."""
        for split_name, examples in splits.items():
            out_path = output_dir / f"{split_name}.jsonl"
            fm.write_jsonl(out_path, examples)
            log.info(f"Saved {split_name}: {len(examples)} examples → {out_path}")

    def _save_hf_dataset(self, splits: Dict[str, List], output_dir: Path) -> None:
        """Save as HuggingFace Dataset (Arrow format)."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            log.error("datasets library not installed. Run: pip install datasets")
            return

        hf_splits = {}
        for split_name, examples in splits.items():
            if examples:
                hf_splits[split_name] = Dataset.from_list(examples)

        dataset_dict = DatasetDict(hf_splits)
        hf_output = output_dir / "hf_dataset"
        dataset_dict.save_to_disk(str(hf_output))
        log.info(f"Saved HuggingFace Dataset → {hf_output}")

        # Also save dataset card
        card = f"""---
language:
- en
license: other
task_categories:
- text-generation
task_ids:
- language-modeling
pretty_name: CodeLlama Fine-Tuning Dataset
---

# CodeLlama Fine-Tuning Dataset

Auto-generated dataset for fine-tuning CodeLlama on programming tasks.

## Splits

| Split | Examples |
|-------|----------|
"""
        for split_name, examples in splits.items():
            card += f"| {split_name} | {len(examples)} |\n"

        card += "\n## Format\n\nEach example contains:\n- `text`: Full formatted prompt+response\n- `instruction`: The task instruction\n- `input`: Optional context\n- `output`: Expected response\n"

        fm.write_text(hf_output / "README.md", card)

    def convert(self, input_dir: Path, output_dir: Path, output_format: str = "both") -> Dict:
        """
        Full conversion pipeline.

        Args:
            input_dir: Directory with training_pairs.jsonl
            output_dir: Where to write converted dataset
            output_format: "jsonl", "hf", or "both"

        Returns:
            Stats dict
        """
        fm._ensure_dir(output_dir)

        # Load
        examples = self._load_examples(input_dir)

        # Analyze lengths
        stats = analyze_lengths(examples[:1000], self.formatter)  # Sample for speed

        # Split
        train, val, test = self._split(examples)

        # Format
        log.info("Applying prompt formatting...")
        formatted_splits = {
            "train": self._format_for_training(train),
            "validation": self._format_for_training(val),
            "test": self._format_for_training(test),
        }

        # Save
        if output_format in ("jsonl", "both"):
            self._save_jsonl_splits(formatted_splits, output_dir)

        if output_format in ("hf", "both"):
            self._save_hf_dataset(formatted_splits, output_dir)

        # Save metadata
        result = {
            "total_examples": len(examples),
            "train": len(formatted_splits["train"]),
            "validation": len(formatted_splits["validation"]),
            "test": len(formatted_splits["test"]),
            "prompt_template": self.formatter.template_name,
            "length_stats": stats,
        }
        fm.write_metadata(output_dir, result)

        log.info("\nConversion complete:")
        for k, v in result.items():
            if k != "length_stats":
                log.info(f"  {k}: {v}")

        return result


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset converter")
    parser.add_argument("--input", required=True, help="Input dir with training_pairs.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for converted dataset")
    parser.add_argument("--format", choices=["jsonl", "hf", "both"], default="both",
                        help="Output format: jsonl, hf (HuggingFace), or both")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    converter = DatasetConverter(config)
    result = converter.convert(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        output_format=args.format
    )
    log.info(f"Done. Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
