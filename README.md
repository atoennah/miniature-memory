# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Philosophy

`miniature-memory` is an experiment in doing more with less. It is built on a "constraint-first" philosophy, designed to train a coherent language model on a continuously growing dataset with minimal hardware (~2GB RAM).

The project's core is its deterministic, auditable, and fully automated data pipeline, which handles everything from web scraping and text extraction to data cleaning and preparation for training.

**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** For a full breakdown of this non-negotiable mission, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Key Features

-   **Automated Data Pipeline:** From Google Search to a training-ready `train.txt`, the entire data workflow is scripted and reproducible.
-   **Constraint-First Design:** Optimized for low-RAM, CPU-only, and free-tier environments.
-   **Incremental Growth:** The dataset is designed to grow continuously without requiring training restarts.
-   **NanoGPT-from-Scratch:** A minimal, understandable, and hackable implementation of a GPT-style model.

## Getting Started

For contributors, the **[Developer Guide](DEVELOPER_GUIDE.md)** is the single source of truth. It provides a comprehensive walkthrough of the environment setup, data pipeline, training commands, and contribution workflow.

### Quick Commands

-   **Install dependencies:** `pip install -r requirements.txt`
-   **Run the full data pipeline:** See the step-by-step commands in the [Developer Guide](DEVELOPER_GUIDE.md#3-the-data-pipeline).
-   **Start training:** `python3 training/train.py --config training/configs/small.yaml`
-   **Generate text:** `python3 scripts/generate.py --max_new_tokens 500`

## Project Documentation

-   **[Developer Guide](DEVELOPER_GUIDE.md):** The essential guide for all contributors. **Start here.**
-   **[CONTRIBUTING.md](CONTRIBUTING.md):** The locked project intent and contribution rules.
-   **[ROADMAP.md](ROADMAP.md):** The long-term vision and development phases.
-   **[DATA_FORMAT.md](DATA_FORMAT.md):** Technical specification for the training data format.
-   **Agent & Memory Files:** The `.jules/` directory contains internal documentation for the AI agents that help maintain this repository.
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

This project is not a general-purpose framework. It is a focused, opinionated, and resource-aware system for building a very specific type of language model.
