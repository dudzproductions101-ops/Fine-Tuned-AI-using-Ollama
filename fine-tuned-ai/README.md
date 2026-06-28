# 🤖 CodeLlama Fine-Tuning Pipeline

A complete, end-to-end pipeline for scraping programming content, building a training dataset, fine-tuning CodeLlama, and deploying it locally via Ollama.

---

## 📁 Project Structure

```
project/
├── scraper/
│   ├── scraper.py          # Web scraping pipeline
│   ├── sources.yaml        # Scraping source configs
│   └── robots_check.py     # robots.txt compliance checker
├── dataset/
│   ├── builder.py          # Dataset construction from raw scraped data
│   ├── preprocessor.py     # Text cleaning, dedup, filtering
│   └── converter.py        # Convert to JSONL / HuggingFace Dataset
├── training/
│   ├── train.py            # Fine-tuning script (LoRA / QLoRA)
│   ├── trainer_config.yaml # Training hyperparameters
│   └── evaluate.py         # Evaluation script
├── model/
│   └── export.py           # Export fine-tuned model for Ollama
├── ollama/
│   └── Modelfile           # Ollama model definition
├── utils/
│   ├── logger.py           # Logging utility
│   ├── file_manager.py     # Auto file/folder creation
│   └── helpers.py          # Shared helpers
├── configs/
│   └── config.yaml         # Global config
├── logs/                   # Auto-generated logs
├── requirements.txt        # All Python dependencies
└── run_pipeline.py         # Full pipeline runner (one command)
```

---

## ⚡ Quick Start (Full Pipeline)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the entire pipeline end-to-end
python run_pipeline.py

# OR run each step individually (see below)
```

---

## 🔧 Step-by-Step Guide

### Step 1 — Scrape Data
```bash
python scraper/scraper.py --output data/raw --max-pages 500
```
Scrapes programming tutorials, docs, Q&A, and examples from configured sources.

### Step 2 — Build & Preprocess Dataset
```bash
python dataset/preprocessor.py --input data/raw --output data/cleaned
python dataset/builder.py --input data/cleaned --output data/dataset
python dataset/converter.py --input data/dataset --output data/final --format jsonl
```

### Step 3 — Fine-Tune
```bash
python training/train.py --config configs/config.yaml
```
Uses QLoRA to fine-tune CodeLlama-7b on your dataset. Requires ~16GB VRAM.
For CPU-only: set `use_4bit: true` and `per_device_train_batch_size: 1` in config.

### Step 4 — Evaluate
```bash
python training/evaluate.py --model outputs/fine_tuned --dataset data/final
```

### Step 5 — Export to Ollama
```bash
python model/export.py --model outputs/fine_tuned --output ollama/model_weights
ollama create mycodellama -f ollama/Modelfile
ollama run mycodellama
```

---

## 🔄 Updating the Dataset & Re-Training

```bash
# Scrape fresh data
python scraper/scraper.py --output data/raw_new --max-pages 200

# Merge with existing
python dataset/builder.py --input data/raw_new --output data/dataset_new --merge data/final

# Re-train from checkpoint
python training/train.py --config configs/config.yaml --resume outputs/fine_tuned
```

---

## 💻 Hardware Requirements

| Mode | RAM | VRAM | Notes |
|------|-----|------|-------|
| Full fine-tune | 32GB | 40GB | A100/H100 |
| QLoRA 4-bit | 16GB | 16GB | RTX 3090/4090 ✅ |
| QLoRA CPU | 32GB | None | Slow but works |

---

## 🛠 Ollama Usage

After setup:
```bash
ollama run mycodellama "Write a Python function to parse JSON"
```

Or with the API:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "mycodellama",
  "prompt": "Write a REST API in FastAPI"
}'
```
