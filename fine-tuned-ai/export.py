"""
model/export.py
---------------
Exports the fine-tuned model for use with Ollama:

  Step 1: Merge LoRA adapter weights into the base model
  Step 2: Save the merged model in safetensors format
  Step 3: Convert to GGUF format (required by Ollama)
  Step 4: Update the Modelfile to point to the new weights

Ollama uses the llama.cpp GGUF format internally.
This script handles the full conversion pipeline.

Requirements:
  - Fine-tuned model (LoRA adapters) at --model path
  - llama.cpp convert script (auto-downloaded if missing)
  - llama.cpp quantize binary (auto-built if missing)

Usage:
    python model/export.py --model outputs/fine_tuned --output ollama/model_weights
    python model/export.py --model outputs/fine_tuned --quant q4_K_M
    python model/export.py --model outputs/merged --skip-merge --output ollama/model_weights
"""

import sys
import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.file_manager import FileManager
from utils.helpers import load_config

log = get_logger("exporter")
fm = FileManager()


# ============================================================
# Step 1: Merge LoRA Weights
# ============================================================

def merge_lora_weights(adapter_path: str, merged_output: str) -> str:
    """
    Merge LoRA adapter weights into the base model.
    This produces a full standalone model that doesn't need PEFT at inference.

    Args:
        adapter_path: Path to the LoRA fine-tuned model directory
        merged_output: Where to save the merged model

    Returns:
        Path to merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    merged_output = Path(merged_output)
    fm._ensure_dir(merged_output)

    # Read adapter config to find base model
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        log.warning("No adapter_config.json found. Assuming already merged model.")
        if adapter_path != merged_output:
            shutil.copytree(str(adapter_path), str(merged_output), dirs_exist_ok=True)
        return str(merged_output)

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("Could not find base_model_name_or_path in adapter_config.json")

    log.info(f"Base model: {base_model_name}")
    log.info(f"LoRA adapters: {adapter_path}")
    log.info(f"Merging to: {merged_output}")

    # Load base model
    log.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter
    log.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # Merge adapter weights into base model
    log.info("Merging LoRA weights (this may take a few minutes)...")
    model = model.merge_and_unload()

    # Save merged model
    log.info(f"Saving merged model to {merged_output}...")
    model.save_pretrained(str(merged_output), safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_output))

    log.info("Merge complete!")
    return str(merged_output)


# ============================================================
# Step 2: Setup llama.cpp
# ============================================================

def setup_llamacpp(llamacpp_dir: str = "llama.cpp") -> Path:
    """
    Clone and build llama.cpp if not already present.
    llama.cpp provides the convert_hf_to_gguf.py script and quantize binary.

    Returns:
        Path to llama.cpp directory
    """
    llamacpp_path = Path(llamacpp_dir)

    if llamacpp_path.exists() and (llamacpp_path / "convert_hf_to_gguf.py").exists():
        log.info(f"llama.cpp found at {llamacpp_path}")
        return llamacpp_path

    log.info("Cloning llama.cpp repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(llamacpp_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone llama.cpp: {result.stderr}")

    log.info("Installing llama.cpp Python requirements...")
    req_file = llamacpp_path / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            check=True
        )

    # Build quantize binary (optional, but needed for quantization)
    log.info("Building llama.cpp quantize binary (may take a few minutes)...")
    build_dir = llamacpp_path / "build"
    build_dir.mkdir(exist_ok=True)

    # Try cmake build
    try:
        subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(build_dir), capture_output=True, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j4"],
            cwd=str(build_dir), capture_output=True, check=True
        )
        log.info("llama.cpp build complete.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("Could not build llama.cpp from source (cmake not found or build failed).")
        log.warning("Quantization will be skipped. Install cmake and try again.")

    return llamacpp_path


# ============================================================
# Step 3: Convert to GGUF
# ============================================================

def convert_to_gguf(merged_model_path: str, output_path: str, llamacpp_dir: str = "llama.cpp") -> str:
    """
    Convert a merged HuggingFace model to GGUF format.
    GGUF is the format used by llama.cpp and Ollama.

    Args:
        merged_model_path: Path to merged model directory
        output_path: Where to save the .gguf file
        llamacpp_dir: Path to llama.cpp repository

    Returns:
        Path to the output .gguf file
    """
    llamacpp_path = Path(llamacpp_dir)
    convert_script = llamacpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        # Try older script name
        convert_script = llamacpp_path / "convert.py"

    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found in {llamacpp_path}. "
            "Run setup_llamacpp() first or provide a valid llama.cpp path."
        )

    output_path = Path(output_path)
    fm._ensure_dir(output_path.parent)

    gguf_file = output_path / "model.gguf"

    log.info(f"Converting to GGUF: {merged_model_path} → {gguf_file}")

    result = subprocess.run(
        [
            sys.executable, str(convert_script),
            str(merged_model_path),
            "--outfile", str(gguf_file),
            "--outtype", "f16",
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log.error(f"GGUF conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed. See logs above.")

    log.info(f"GGUF file created: {gguf_file}")
    return str(gguf_file)


# ============================================================
# Step 4: Quantize GGUF
# ============================================================

def quantize_gguf(
    gguf_file: str,
    output_dir: str,
    quant_format: str = "q4_K_M",
    llamacpp_dir: str = "llama.cpp"
) -> str:
    """
    Quantize the GGUF file to reduce size.
    Common formats:
      - q4_K_M: 4-bit quantization, medium quality (~4GB for 7B model)
      - q5_K_M: 5-bit, better quality (~4.5GB)
      - q8_0:   8-bit, near lossless (~7GB)
      - f16:    Full float16, no loss (~14GB)

    Args:
        gguf_file: Input .gguf file path
        output_dir: Directory to save quantized file
        quant_format: Quantization format string
        llamacpp_dir: Path to llama.cpp with quantize binary

    Returns:
        Path to quantized .gguf file
    """
    llamacpp_path = Path(llamacpp_dir)

    # Find quantize binary
    quantize_bin = None
    candidates = [
        llamacpp_path / "build" / "bin" / "llama-quantize",
        llamacpp_path / "build" / "bin" / "quantize",
        llamacpp_path / "quantize",
        llamacpp_path / "llama-quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            quantize_bin = candidate
            break

    if not quantize_bin:
        log.warning(
            "llama-quantize binary not found. Skipping quantization.\n"
            "The unquantized f16 GGUF will be used instead.\n"
            "To quantize, build llama.cpp: cd llama.cpp && cmake -B build && cmake --build build -j"
        )
        return str(gguf_file)

    output_dir = Path(output_dir)
    fm._ensure_dir(output_dir)
    quantized_file = output_dir / f"model-{quant_format}.gguf"

    log.info(f"Quantizing: {gguf_file} → {quantized_file} ({quant_format})")

    result = subprocess.run(
        [str(quantize_bin), str(gguf_file), str(quantized_file), quant_format.upper()],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        log.error(f"Quantization failed:\n{result.stderr}")
        log.warning("Falling back to unquantized GGUF.")
        return str(gguf_file)

    file_size = quantized_file.stat().st_size / 1e9
    log.info(f"Quantized model: {quantized_file} ({file_size:.1f} GB)")
    return str(quantized_file)


# ============================================================
# Step 5: Update Modelfile
# ============================================================

def update_modelfile(gguf_path: str, modelfile_path: str = "ollama/Modelfile") -> None:
    """Update the Ollama Modelfile to point to the new GGUF file."""
    modelfile = Path(modelfile_path)

    if not modelfile.exists():
        log.warning(f"Modelfile not found at {modelfile_path}. Skipping update.")
        return

    content = modelfile.read_text()

    # Replace FROM line
    import re
    content = re.sub(
        r"^FROM\s+.*$",
        f"FROM {gguf_path}",
        content,
        flags=re.MULTILINE
    )

    modelfile.write_text(content)
    log.info(f"Updated Modelfile FROM to: {gguf_path}")


# ============================================================
# Full Export Pipeline
# ============================================================

def export(
    model_path: str,
    output_dir: str,
    quant_format: str = "q4_K_M",
    skip_merge: bool = False,
    modelfile_path: str = "ollama/Modelfile",
    llamacpp_dir: str = "llama.cpp",
) -> Dict:
    """
    Full export pipeline: merge → convert → quantize → update Modelfile.

    Args:
        model_path: Path to fine-tuned model (with LoRA adapters)
        output_dir: Base output directory
        quant_format: GGUF quantization format
        skip_merge: If True, assume model_path is already merged
        modelfile_path: Path to Ollama Modelfile to update
        llamacpp_dir: Path to llama.cpp

    Returns:
        Dict with paths to all generated artifacts
    """
    output_dir = Path(output_dir)
    fm._ensure_dir(output_dir)

    results = {
        "model_path": model_path,
        "output_dir": str(output_dir),
    }

    # Step 1: Merge LoRA
    if skip_merge:
        merged_path = model_path
        log.info(f"Skipping merge. Using: {merged_path}")
    else:
        merged_dir = output_dir.parent.parent / "outputs" / "merged"
        log.info("\n[Step 1/4] Merging LoRA weights...")
        merged_path = merge_lora_weights(model_path, str(merged_dir))
        results["merged_model"] = merged_path

    # Step 2: Setup llama.cpp
    log.info("\n[Step 2/4] Setting up llama.cpp...")
    try:
        llamacpp_path = setup_llamacpp(llamacpp_dir)
    except Exception as e:
        log.error(f"llama.cpp setup failed: {e}")
        log.error("Please install llama.cpp manually: https://github.com/ggerganov/llama.cpp")
        return results

    # Step 3: Convert to GGUF
    log.info("\n[Step 3/4] Converting to GGUF...")
    try:
        gguf_file = convert_to_gguf(merged_path, str(output_dir), str(llamacpp_path))
        results["gguf_file"] = gguf_file
    except Exception as e:
        log.error(f"GGUF conversion failed: {e}")
        return results

    # Step 4: Quantize
    log.info("\n[Step 4/4] Quantizing GGUF...")
    final_gguf = quantize_gguf(gguf_file, str(output_dir), quant_format, str(llamacpp_path))
    results["final_gguf"] = final_gguf

    # Step 5: Update Modelfile
    update_modelfile(final_gguf, modelfile_path)
    results["modelfile"] = modelfile_path

    log.info(f"\n{'='*50}")
    log.info("EXPORT COMPLETE")
    log.info(f"Final GGUF: {final_gguf}")
    log.info(f"\nNext steps:")
    log.info(f"  1. ollama create mycodellama -f {modelfile_path}")
    log.info(f"  2. ollama run mycodellama")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to Ollama")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model (LoRA adapters)")
    parser.add_argument("--output", default="ollama/model_weights", help="Output directory")
    parser.add_argument("--quant", default="q4_K_M", help="GGUF quantization format")
    parser.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge (already merged)")
    parser.add_argument("--modelfile", default="ollama/Modelfile", help="Ollama Modelfile path")
    parser.add_argument("--llamacpp", default="llama.cpp", help="llama.cpp directory")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    results = export(
        model_path=args.model,
        output_dir=args.output,
        quant_format=args.quant,
        skip_merge=args.skip_merge,
        modelfile_path=args.modelfile,
        llamacpp_dir=args.llamacpp,
    )

    log.info("\nExport Results:")
    for k, v in results.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
