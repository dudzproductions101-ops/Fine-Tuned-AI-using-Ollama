"""
training/evaluate.py
--------------------
Evaluates the fine-tuned model on the test set using multiple metrics:
  - Perplexity: How well the model predicts the next token (lower = better)
  - ROUGE scores: Text overlap between generated and reference outputs
  - Code quality heuristics: Does output look like valid code?
  - Pass@k: (For coding tasks) Ratio of syntactically correct outputs

Also supports interactive inference for manual testing.

Usage:
    python training/evaluate.py --model outputs/fine_tuned --dataset data/final
    python training/evaluate.py --model outputs/fine_tuned --dataset data/final --interactive
    python training/evaluate.py --model outputs/fine_tuned --sample 50
"""

import sys
import json
import math
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config

log = get_logger("evaluator")
fm = FileManager()


# ============================================================
# Model Loader for Evaluation
# ============================================================

def load_model_for_eval(model_path: str, device: str = "auto"):
    """
    Load fine-tuned model (with LoRA adapters merged) for inference.
    Much simpler than training loading — no quantization needed for eval.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info(f"Loading model from: {model_path}")
    model_path = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a LoRA adapter or merged model
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        log.info("Detected LoRA adapter. Loading base + adapter...")
        with open(adapter_config) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "codellama/CodeLlama-7b-Instruct-hf")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    else:
        log.info("Loading merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


# ============================================================
# Metrics
# ============================================================

class Perplexity:
    """Compute perplexity over the test set."""

    def __init__(self, model, tokenizer, max_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute(self, texts: List[str]) -> float:
        """Compute average perplexity over a list of texts."""
        total_loss = 0.0
        total_tokens = 0

        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(self.device)

            if input_ids.shape[1] < 2:
                continue

            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.shape[1] - 1  # Shifted by 1 for language modeling

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)


class RougeScorer:
    """Compute ROUGE-L scores for generated vs reference text."""

    def __init__(self):
        try:
            from rouge_score import rouge_scorer as rs
            self.scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            self.available = True
        except ImportError:
            log.warning("rouge_score not available. ROUGE metrics will be skipped.")
            self.available = False

    def score(self, prediction: str, reference: str) -> Dict:
        """Score a single prediction against a reference."""
        if not self.available:
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

        scores = self.scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def batch_score(self, predictions: List[str], references: List[str]) -> Dict:
        """Average ROUGE scores over a batch."""
        if not predictions:
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

        aggregated = defaultdict(list)
        for pred, ref in zip(predictions, references):
            s = self.score(pred, ref)
            for k, v in s.items():
                aggregated[k].append(v)

        return {k: sum(v) / len(v) for k, v in aggregated.items()}


class CodeQualityChecker:
    """
    Heuristic checks for code quality in model outputs.
    Checks syntax validity for Python (using ast.parse).
    """

    def check_python_syntax(self, code: str) -> bool:
        """Check if Python code is syntactically valid."""
        import ast
        # Extract code from markdown if needed
        code = re.sub(r"```python\n?", "", code)
        code = re.sub(r"```\n?", "", code)
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def has_code_structure(self, text: str) -> bool:
        """Check if output contains code-like structure."""
        patterns = [
            r"def\s+\w+\s*\(",
            r"class\s+\w+",
            r"import\s+\w+",
            r"```",
            r"function\s+\w+",
            r"const\s+\w+\s*=",
        ]
        return any(re.search(p, text) for p in patterns)

    def estimate_quality_score(self, output: str, expected_code: bool = True) -> float:
        """
        Estimate output quality (0–1 scale).
        Combines multiple heuristics.
        """
        score = 0.0
        weights = 0.0

        # Length check
        if len(output) > 20:
            score += 0.3
        weights += 0.3

        # Code structure (if coding task)
        if expected_code:
            if self.has_code_structure(output):
                score += 0.4
            weights += 0.4

        # Python syntax (if looks like Python)
        if "def " in output or "import " in output:
            if self.check_python_syntax(output):
                score += 0.3
            weights += 0.3

        return score / weights if weights > 0 else 0.0


# ============================================================
# Generator for Evaluation
# ============================================================

class ModelGenerator:
    """Wrapper for generating completions from the fine-tuned model."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.2, top_p: float = 0.95) -> str:
        """Generate a completion for a prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate completions for multiple prompts."""
        return [self.generate(p, **kwargs) for p in tqdm(prompts, desc="Generating")]


# ============================================================
# Evaluation Pipeline
# ============================================================

class EvaluationPipeline:
    """Runs the full evaluation suite on the test set."""

    def __init__(self, model_path: str, dataset_path: str, config: Dict):
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.max_length = config["model"].get("max_length", 2048)

    def load_test_examples(self, n_samples: Optional[int] = None) -> List[Dict]:
        """Load test examples from the dataset."""
        test_file = self.dataset_path / "test.jsonl"
        hf_path = self.dataset_path / "hf_dataset"

        if test_file.exists():
            examples = fm.read_jsonl(test_file)
        elif hf_path.exists():
            from datasets import load_from_disk
            ds = load_from_disk(str(hf_path))
            if "test" in ds:
                examples = [dict(row) for row in ds["test"]]
            else:
                examples = [dict(row) for row in ds["validation"]]
        else:
            raise FileNotFoundError(f"No test data found in {self.dataset_path}")

        if n_samples and n_samples < len(examples):
            import random
            random.shuffle(examples)
            examples = examples[:n_samples]

        log.info(f"Loaded {len(examples)} test examples")
        return examples

    def run(self, n_samples: Optional[int] = None) -> Dict:
        """
        Run full evaluation.

        Returns:
            Dict of all computed metrics
        """
        log.info(f"{'='*50}")
        log.info("EVALUATION PIPELINE")
        log.info(f"Model: {self.model_path}")
        log.info(f"Dataset: {self.dataset_path}")

        # Load model
        model, tokenizer = load_model_for_eval(self.model_path)
        generator = ModelGenerator(model, tokenizer)
        rouge = RougeScorer()
        code_checker = CodeQualityChecker()

        # Load test examples
        examples = self.load_test_examples(n_samples)

        # ---- 1. Perplexity ----
        log.info("\n[1/3] Computing perplexity on test set...")
        ppl_calc = Perplexity(model, tokenizer, self.max_length)
        texts = [ex.get("text", "") for ex in examples[:100]]  # Sample for speed
        perplexity = ppl_calc.compute(texts)
        log.info(f"  Perplexity: {perplexity:.2f}")

        # ---- 2. Generation Quality ----
        log.info("\n[2/3] Evaluating generation quality...")

        predictions = []
        references = []
        quality_scores = []

        for ex in tqdm(examples[:min(50, len(examples))], desc="Generating outputs"):
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            reference = ex.get("output", "")

            # Build prompt (without answer)
            if input_text:
                prompt = f"[INST] {instruction}\n\n### Input:\n{input_text} [/INST]"
            else:
                prompt = f"[INST] {instruction} [/INST]"

            prediction = generator.generate(prompt)
            predictions.append(prediction)
            references.append(reference)

            # Quality score
            is_code_task = any(kw in instruction.lower() for kw in
                               ["write", "implement", "create", "code", "function", "class"])
            quality_scores.append(code_checker.estimate_quality_score(prediction, is_code_task))

        # ---- 3. ROUGE Scores ----
        log.info("\n[3/3] Computing ROUGE scores...")
        rouge_scores = rouge.batch_score(predictions, references)

        # ---- Compile Results ----
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        results = {
            "model_path": str(self.model_path),
            "n_test_examples": len(examples),
            "n_evaluated": len(predictions),
            "perplexity": round(perplexity, 4),
            "rouge1": round(rouge_scores["rouge1"], 4),
            "rouge2": round(rouge_scores["rouge2"], 4),
            "rougeL": round(rouge_scores["rougeL"], 4),
            "avg_quality_score": round(avg_quality, 4),
        }

        # ---- Print Results ----
        log.info(f"\n{'='*50}")
        log.info("EVALUATION RESULTS")
        log.info(f"{'='*50}")
        for k, v in results.items():
            log.info(f"  {k}: {v}")
        log.info(f"{'='*50}")

        # Show sample predictions
        log.info("\nSample Predictions (first 3):")
        for i, (pred, ref, ex) in enumerate(zip(predictions[:3], references[:3], examples[:3])):
            log.info(f"\n--- Example {i+1} ---")
            log.info(f"Instruction: {ex.get('instruction', '')[:100]}")
            log.info(f"Reference:   {ref[:200]}...")
            log.info(f"Predicted:   {pred[:200]}...")

        # Save results
        results_path = Path(self.model_path) / "eval_results.json"
        fm.write_json(results_path, results)
        log.info(f"\nResults saved to: {results_path}")

        return results


# ============================================================
# Interactive Mode
# ============================================================

def interactive_mode(model_path: str):
    """Run interactive inference loop for manual testing."""
    model, tokenizer = load_model_for_eval(model_path)
    generator = ModelGenerator(model, tokenizer, max_new_tokens=512)

    log.info("\nInteractive mode. Type 'quit' to exit.\n")

    while True:
        instruction = input("\nInstruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        input_text = input("Input (optional, press Enter to skip): ").strip()

        if input_text:
            prompt = f"[INST] {instruction}\n\n### Input:\n{input_text} [/INST]"
        else:
            prompt = f"[INST] {instruction} [/INST]"

        print("\nGenerating...\n")
        output = generator.generate(prompt)
        print(f"Output:\n{output}\n")
        print("-" * 50)


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset", default="data/final", help="Path to dataset directory")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Number of test samples to evaluate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.interactive:
        interactive_mode(args.model)
    else:
        pipeline = EvaluationPipeline(args.model, args.dataset, config)
        results = pipeline.run(n_samples=args.sample)
        log.info(f"Evaluation complete. Perplexity: {results['perplexity']:.2f}")


if __name__ == "__main__":
    main()
