# Developer Guide

This guide provides a technical overview of the miniature-memory project, including repository structure, setup, and the training workflow.

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
