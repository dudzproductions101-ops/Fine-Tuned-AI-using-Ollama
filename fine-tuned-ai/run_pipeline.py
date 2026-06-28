"""
run_pipeline.py
---------------
Master pipeline runner. Orchestrates all steps end-to-end:

  1. Web Scraping     → data/raw/
  2. Preprocessing    → data/cleaned/
  3. Dataset Building → data/dataset/
  4. Dataset Convert  → data/final/
  5. Fine-Tuning      → outputs/fine_tuned/
  6. Evaluation       → outputs/fine_tuned/eval_results.json
  7. Export to Ollama → ollama/model_weights/

Usage:
    # Full pipeline (all steps)
    python run_pipeline.py

    # Start from a specific step
    python run_pipeline.py --start-step preprocess
    python run_pipeline.py --start-step train

    # Skip export (just train)
    python run_pipeline.py --skip export

    # Use a specific config
    python run_pipeline.py --config configs/config.yaml

    # Dry run (show what would happen)
    python run_pipeline.py --dry-run
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger, get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config

log = get_logger("pipeline")
fm = FileManager()


# ============================================================
# Pipeline Steps Definition
# ============================================================

STEPS = [
    "scrape",
    "preprocess",
    "build",
    "convert",
    "train",
    "evaluate",
    "export",
]


# ============================================================
# Step Implementations
# ============================================================

def run_step(step_name: str, config: Dict, dry_run: bool = False) -> bool:
    """
    Run a single pipeline step.

    Returns:
        True if step succeeded, False if failed.
    """
    log.info(f"\n{'='*60}")
    log.info(f"STEP: {step_name.upper()}")
    log.info(f"{'='*60}")

    if dry_run:
        log.info(f"[DRY RUN] Would run: {step_name}")
        return True

    start_time = time.time()
    success = False

    try:
        if step_name == "scrape":
            success = step_scrape(config)
        elif step_name == "preprocess":
            success = step_preprocess(config)
        elif step_name == "build":
            success = step_build(config)
        elif step_name == "convert":
            success = step_convert(config)
        elif step_name == "train":
            success = step_train(config)
        elif step_name == "evaluate":
            success = step_evaluate(config)
        elif step_name == "export":
            success = step_export(config)
        else:
            log.error(f"Unknown step: {step_name}")
            return False

    except KeyboardInterrupt:
        log.warning(f"\nStep '{step_name}' interrupted by user.")
        return False
    except Exception as e:
        log.error(f"Step '{step_name}' failed with exception: {e}", exc_info=True)
        return False

    elapsed = time.time() - start_time
    status = "✓ SUCCESS" if success else "✗ FAILED"
    log.info(f"\n{status} — {step_name} ({elapsed:.1f}s)")
    return success


def step_scrape(config: Dict) -> bool:
    """Run web scraping pipeline."""
    from scraper.scraper import ScraperPipeline
    import yaml

    with open("scraper/sources.yaml") as f:
        sources_config = yaml.safe_load(f)

    sources = sources_config.get("sources", [])
    output_dir = config.get("scraper", {}).get("output_dir", "data/raw")
    fm._ensure_dir(output_dir)

    pipeline = ScraperPipeline(config, sources)
    results = pipeline.run()

    total = sum(results.values())
    log.info(f"Scraping complete: {total} total records across {len(results)} sources")
    return total > 0


def step_preprocess(config: Dict) -> bool:
    """Run text preprocessing and deduplication."""
    from dataset.preprocessor import PreprocessingPipeline

    raw_dir = Path(config.get("scraper", {}).get("output_dir", "data/raw"))
    cleaned_dir = Path("data/cleaned")

    if not any(raw_dir.glob("*.jsonl")):
        log.warning(f"No JSONL files in {raw_dir}. Run 'scrape' step first.")
        log.info("Creating a small sample dataset for testing...")
        _create_sample_data(raw_dir)

    pipeline = PreprocessingPipeline(config)
    total = pipeline.process_directory(raw_dir, cleaned_dir)

    log.info(f"Preprocessing complete: {total} records kept")
    return total > 0


def step_build(config: Dict) -> bool:
    """Run dataset builder to create instruction/output pairs."""
    from dataset.builder import DatasetBuilder

    cleaned_dir = Path("data/cleaned")
    dataset_dir = Path("data/dataset")

    if not any(cleaned_dir.glob("*.jsonl")):
        log.warning(f"No cleaned data in {cleaned_dir}. Run 'preprocess' step first.")
        return False

    builder = DatasetBuilder(config)
    total = builder.build(
        input_dir=cleaned_dir,
        output_dir=dataset_dir
    )

    log.info(f"Dataset build complete: {total} examples")
    return total > 0


def step_convert(config: Dict) -> bool:
    """Convert dataset to HF format with train/val/test splits."""
    from dataset.converter import DatasetConverter

    dataset_dir = Path("data/dataset")
    final_dir = Path("data/final")

    if not any(dataset_dir.glob("*.jsonl")):
        log.warning(f"No dataset in {dataset_dir}. Run 'build' step first.")
        return False

    converter = DatasetConverter(config)
    result = converter.convert(
        input_dir=dataset_dir,
        output_dir=final_dir,
        output_format="both"
    )

    total = result.get("total_examples", 0)
    log.info(f"Conversion complete: {total} total examples")
    return total > 0


def step_train(config: Dict) -> bool:
    """Run fine-tuning."""
    from training.train import train

    dataset_path = "data/final"
    output_dir = config["training"]["output_dir"]

    if not Path(dataset_path).exists():
        log.warning(f"Dataset not found at {dataset_path}. Run 'convert' step first.")
        return False

    log.info("Starting fine-tuning...")
    log.info("This will take significant time depending on your hardware.")
    log.info(f"Checkpoints will be saved to: {output_dir}")

    metrics = train(
        config=config,
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    final_loss = metrics.get("train_loss", "N/A")
    log.info(f"Training complete. Final loss: {final_loss}")
    return True


def step_evaluate(config: Dict) -> bool:
    """Run evaluation on the test set."""
    from training.evaluate import EvaluationPipeline

    model_path = config["training"]["output_dir"]
    dataset_path = "data/final"

    if not Path(model_path).exists():
        log.warning(f"No trained model at {model_path}. Run 'train' step first.")
        return False

    pipeline = EvaluationPipeline(model_path, dataset_path, config)
    results = pipeline.run(n_samples=50)  # Use 50 samples for speed

    log.info(f"Evaluation complete:")
    log.info(f"  Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    log.info(f"  ROUGE-L: {results.get('rougeL', 'N/A'):.4f}")
    return True


def step_export(config: Dict) -> bool:
    """Export model to Ollama format."""
    from model.export import export

    model_path = config["training"]["output_dir"]
    output_dir = config["export"]["ollama_weights_dir"]
    quant = config["export"].get("quantization_format", "q4_K_M")

    if not Path(model_path).exists():
        log.warning(f"No trained model at {model_path}. Run 'train' step first.")
        return False

    log.info(f"Exporting model to Ollama (quantization: {quant})...")
    results = export(
        model_path=model_path,
        output_dir=output_dir,
        quant_format=quant,
        modelfile_path=config["export"]["modelfile_path"],
    )

    if "final_gguf" in results:
        log.info(f"\nExport success! GGUF: {results['final_gguf']}")
        _print_ollama_instructions(config)
        return True

    return False


# ============================================================
# Helpers
# ============================================================

def _create_sample_data(output_dir: Path) -> None:
    """
    Create a tiny sample dataset for testing the pipeline
    without running the actual scraper.
    """
    fm._ensure_dir(output_dir)
    sample_file = output_dir / "sample_20240101_000000.jsonl"

    import json, hashlib
    from datetime import datetime

    samples = [
        {
            "id": "sample_001",
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "sample",
            "content_type": "tutorial",
            "url": "https://example.com/python-tutorial",
            "title": "Python Functions Tutorial",
            "text": """Python functions are reusable blocks of code that perform specific tasks.
You define a function using the def keyword, followed by the function name and parentheses.

def greet(name):
    \"\"\"Return a greeting message.\"\"\"
    return f"Hello, {name}!"

result = greet("World")
print(result)  # Hello, World!

Functions can accept parameters and return values. They help organize code
and avoid repetition. Python functions support default arguments, keyword
arguments, and variable-length argument lists (*args and **kwargs).

def calculate(a, b, operation="add"):
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    raise ValueError(f"Unknown operation: {operation}")
""",
            "code_blocks": [
                'def greet(name):\n    """Return a greeting."""\n    return f"Hello, {name}!"',
                'def calculate(a, b, operation="add"):\n    if operation == "add":\n        return a + b'
            ]
        },
        {
            "id": "sample_002",
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "sample",
            "content_type": "qa",
            "url": "https://stackoverflow.com/sample",
            "title": "How do I sort a list of dictionaries in Python?",
            "text": "I have a list of dictionaries and want to sort them by a specific key value.",
            "answers": [
                {
                    "text": "Use the sorted() function with a key parameter:\n\ndata = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nsorted_data = sorted(data, key=lambda x: x['age'])\n\nOr use itemgetter for slightly better performance:\nfrom operator import itemgetter\nsorted_data = sorted(data, key=itemgetter('age'))",
                    "score": 150,
                    "is_accepted": True
                }
            ],
            "tags": ["python", "sorting", "dictionary"],
            "score": 200,
            "code_blocks": ["sorted(data, key=lambda x: x['age'])"]
        },
        {
            "id": "sample_003",
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "sample",
            "content_type": "documentation",
            "url": "https://docs.python.org/3/library/json.html",
            "title": "json — JSON encoder and decoder",
            "text": """The json module provides methods to encode and decode JSON data in Python.

import json

# Encoding Python object to JSON string
data = {"name": "Alice", "age": 30, "hobbies": ["reading", "coding"]}
json_string = json.dumps(data, indent=2)
print(json_string)

# Decoding JSON string to Python object
parsed = json.loads(json_string)
print(parsed["name"])  # Alice

# Reading from file
with open("data.json", "r") as f:
    data = json.load(f)

# Writing to file
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)

The json.dumps() function accepts several optional arguments:
- indent: Number of spaces for pretty printing
- sort_keys: Sort dictionary keys alphabetically
- ensure_ascii: If False, allows non-ASCII characters
""",
            "code_blocks": [
                'import json\ndata = {"name": "Alice"}\njson_string = json.dumps(data, indent=2)',
                'with open("data.json", "r") as f:\n    data = json.load(f)'
            ]
        }
    ]

    # Add more varied samples
    for i in range(20):
        samples.append({
            "id": f"sample_{i+10:03d}",
            "scraped_at": datetime.utcnow().isoformat(),
            "source": "sample",
            "content_type": "tutorial",
            "url": f"https://example.com/tutorial-{i}",
            "title": f"Programming Tutorial {i}",
            "text": f"""This is a programming tutorial about topic {i}.

Code example {i}:

def example_function_{i}(x, y):
    \"\"\"Compute a result from x and y.\"\"\"
    result = x + y * {i + 1}
    return result

# Usage
value = example_function_{i}(10, 5)
print(f"Result: {{value}}")

The function above demonstrates basic arithmetic operations.
It takes two parameters and returns their combination.
Understanding functions is fundamental to programming.
""",
            "code_blocks": [
                f'def example_function_{i}(x, y):\n    result = x + y * {i+1}\n    return result'
            ]
        })

    with open(sample_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    log.info(f"Created {len(samples)} sample records at {sample_file}")


def _print_ollama_instructions(config: Dict) -> None:
    """Print the steps to register and run the model in Ollama."""
    model_name = config.get("ollama", {}).get("model_name", "mycodellama")
    modelfile = config.get("export", {}).get("modelfile_path", "ollama/Modelfile")

    log.info(f"""
{'='*60}
OLLAMA SETUP INSTRUCTIONS
{'='*60}

1. Install Ollama (if not already):
   https://ollama.com/download

2. Create your model:
   ollama create {model_name} -f {modelfile}

3. Run it:
   ollama run {model_name}

4. Or use the API:
   curl http://localhost:11434/api/generate -d '{{
     "model": "{model_name}",
     "prompt": "Write a Python function to parse JSON"
   }}'

5. Delete model if needed:
   ollama rm {model_name}

{'='*60}
""")


# ============================================================
# Main Pipeline Orchestrator
# ============================================================

def run_pipeline(
    config_path: str = "configs/config.yaml",
    start_step: str = "scrape",
    skip_steps: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    """
    Run the full pipeline from start_step onwards.

    Args:
        config_path: Path to config YAML
        start_step: Which step to start from
        skip_steps: List of step names to skip
        dry_run: If True, show what would run without running

    Returns:
        True if all steps succeeded, False otherwise
    """
    setup_logger(log_dir="logs", log_name="pipeline")

    config = load_config(config_path)
    skip_steps = skip_steps or []

    # Ensure all data directories exist
    fm.ensure_data_dirs()

    # Determine which steps to run
    if start_step not in STEPS:
        log.error(f"Unknown start step: {start_step}. Choose from: {STEPS}")
        return False

    start_idx = STEPS.index(start_step)
    steps_to_run = [s for s in STEPS[start_idx:] if s not in skip_steps]

    log.info(f"""
{'='*60}
CODELLAMA FINE-TUNING PIPELINE
{'='*60}
Config:    {config_path}
Base model: {config['model']['base_model']}
Start step: {start_step}
Steps:     {' → '.join(steps_to_run)}
Dry run:   {dry_run}
{'='*60}
""")

    pipeline_start = time.time()
    results = {}

    for step in steps_to_run:
        success = run_step(step, config, dry_run=dry_run)
        results[step] = success

        if not success:
            log.error(f"Pipeline halted at step '{step}' (failed).")
            log.info("You can resume from this step with:")
            log.info(f"  python run_pipeline.py --start-step {step}")
            break

    # Summary
    elapsed = time.time() - pipeline_start
    log.info(f"\n{'='*60}")
    log.info("PIPELINE SUMMARY")
    log.info(f"Total time: {elapsed/60:.1f} minutes")
    for step, success in results.items():
        status = "✓" if success else "✗"
        log.info(f"  {status} {step}")

    all_success = all(results.values())
    if all_success:
        log.info("\n✓ Pipeline completed successfully!")
    else:
        log.info("\n✗ Pipeline completed with errors.")

    return all_success


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full CodeLlama fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                        # Full pipeline
  python run_pipeline.py --start-step train     # Start from training
  python run_pipeline.py --skip export          # Skip Ollama export
  python run_pipeline.py --dry-run              # Show what would run
        """
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--start-step",
        choices=STEPS,
        default="scrape",
        help=f"Step to start from. Choices: {STEPS}"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=STEPS,
        default=[],
        help="Steps to skip"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show steps without running")
    return parser.parse_args()


def main():
    args = parse_args()
    success = run_pipeline(
        config_path=args.config,
        start_step=args.start_step,
        skip_steps=args.skip,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
