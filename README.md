# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models

miniature-memory is a project focused on building, growing, and training a small text-based language model using a NanoGPT-from-scratch approach, optimized for tight resource constraints (e.g. Google Colab, ~2GB RAM environments).

The core of this project is the dataset pipeline: scraping, extracting, storing, cleaning, preparing, and incrementally training on text data in a fully automated way.

## What This Project Does

- Scrapes text content from the web using Google Search as an entry point
- Extracts readable text using browser-style “read mode”
- Stores raw text as an ever-growing dataset
- Cleans and prepares data for training
- Trains a small GPT-style model incrementally
- Saves checkpoints and artifacts so training can resume anytime
- Designed to run on Linux and Google Colab

This project is not a large-model framework.
It is intentionally small, explicit, and constrained.

## Developer Documentation

This README provides a high-level overview. For a detailed technical explanation of the data pipeline, training process, and project philosophy, please see the **[Developer Guide](DEVELOPER_GUIDE.md)**.

## Target Model

- **Architecture:** GPT-style decoder-only transformer
- **Base implementation:** NanoGPT-style (from scratch)
- **Scale:** very small models
- **Training:** incremental, resumable
- **Environment:** CPU / small GPU

The goal is to push NanoGPT as far as possible under minimal resources.

## Repository Structure

```
.
├── dataset/
│   ├── raw/          # raw extracted text (one file per content)
│   ├── cleaned/      # cleaned text (derived)
│   └── processed/    # tokenized / training-ready data
│
├── scraper/
│   ├── search/       # Google search helpers
│   ├── fetch/        # URL fetching & rendering
│   └── extract/      # read-mode text extraction
│
├── scripts/
│   ├── validate_raw.py
│   ├── clean_dataset.py
│   ├── prepare_data.py
│   └── stats.py
│
├── training/
│   ├── model.py
│   ├── train.py
│   └── configs/
│
├── run.py
├── setup.sh
├── NanoGPT_Training.ipynb
└── README.md
```

## Setup

To set up the environment, run the `setup.sh` script:
```bash
./setup.sh
```
This will install all the necessary dependencies.

## Training Workflow

The entire data processing and training pipeline can be run using the unified `run.py` script.

### Local / Linux

To run the entire pipeline, simply execute:
```bash
python run.py
```

You can also skip specific steps using command-line flags:
```bash
# Skip validation and cleaning
python run.py --skip-validation --skip-cleaning

# Run only the training step
python run.py --skip-validation --skip-cleaning --skip-preparation
```

### Google Colab

- Open `NanoGPT_Training.ipynb`
- Upload or clone repository
- Run all cells top to bottom
- Training artifacts are saved periodically

## Dataset Design

### Raw Dataset

- Raw data is plain text only
- One file = one content item
- Stored exactly as extracted
- Never edited after being written

Example:
`dataset/raw/source_example/20250112_231455__SRC__abc123__9f3a.txt`

Raw data exists only to be consumed by scripts, not humans.

### Cleaned Dataset

- Generated from raw data
- Removes noise (HTML artifacts, encoding issues)
- Normalizes whitespace
- Still text-only

### Processed Dataset

- Tokenized
- Split into train / validation
- Ready for model training

All cleaned and processed data can be re-generated from raw data at any time.

## Scraping Pipeline (High Level)

1.  Search using Google
2.  Collect candidate URLs
3.  Render page (headless / read-mode)
4.  Extract readable text
5.  Save text into `dataset/raw/`
6.  Log source and timestamp

Scraping is designed to be repeatable and incremental.

## Resource Constraints

Designed to work with:

- ~2GB RAM (minimum target)
- Limited disk
- Free-tier Colab environments

Techniques used:

- small vocab
- small context window
- small batch sizes
- checkpoint-based training

## Incremental Growth

- New raw data can be added at any time
- Cleaning and preparation re-run deterministically
- Training resumes from latest checkpoint
- Dataset grows continuously without restarting from zero

## Current Status

- Dataset pipeline: in progress
- Scraping automation: active development
- Training loop: functional
- Colab support: draft / usable

See `ROADMAP.md` for what’s next.

## Summary

miniature-memory is about doing more with less:

- small model
- small machine
- growing dataset
- full control over every step

No black boxes, no magic, no assumptions.

