# Developer Guide: miniature-memory

This guide provides a comprehensive overview of the miniature-memory project, its goals, architecture, and development roadmap. It is intended to be the single source of truth for all technical and strategic information.

## 1. Project Vision & Purpose

### Core Mission
miniature-memory is a project focused on building, growing, and training a small text-based language model using a NanoGPT-from-scratch approach, optimized for tight resource constraints (e.g., Google Colab, ~2GB RAM environments).

The core of this project is the dataset pipeline: scraping, extracting, storing, cleaning, preparing, and incrementally training on text data in a fully automated way.

### Non-Negotiable Intent
This project exists explicitly and exclusively to build, study, and deploy language models for **adult entertainment**. The training datasets, model architecture, and outputs are intended for adult-oriented text generation. This is a locked project intent, and any contributions that attempt to sanitize or redirect this purpose will be rejected. For full details, see the `CONTRIBUTING.md` file.

### Target Model
- **Architecture:** GPT-style decoder-only transformer
- **Base implementation:** NanoGPT-style (from scratch)
- **Scale:** Very small models (typically under 25M parameters)
- **Training:** Incremental, resumable
- **Environment:** CPU / small GPU

The goal is to push NanoGPT as far as possible under minimal resources.

## 2. Repository Structure

The repository is organized to maintain a strict separation of concerns between data, scraping, processing, and training code.

```
.
├── dataset/
│   ├── raw/          # Append-only raw extracted text (one file per content item)
│   ├── cleaned/      # Cleaned, normalized text (derived from raw)
│   ├── processed/    # Tokenized, training-ready data
│   ├── rejected/     # Raw content that failed quality checks
│   └── metadata/     # URL manifests and scraping metadata
│
├── scraper/
│   ├── search/       # Google search helpers
│   ├── fetch/        # URL fetching & rendering
│   └── extract/      # Read-mode text extraction
│
├── processing/
│   ├── normalize.py
│   ├── segment.py
│   └── quality_filter.py
│
├── training/
│   ├── model.py      # The core NanoGPT model architecture
│   ├── train.py      # The training loop
│   └── configs/      # YAML configuration files for different model sizes
│
├── scripts/
│   └── cli.py        # Command-line interface for pipeline operations
│
├── run.py            # Main orchestrator script for the entire pipeline
├── setup.sh          # Environment setup script
└── DEVELOPER_GUIDE.md # This file
```

## 3. Setup & Workflow

### Environment Setup
To set up the environment and install all necessary dependencies, run the `setup.sh` script:
```bash
./setup.sh
```

### End-to-End Pipeline
The entire data processing and training pipeline can be run using the unified `run.py` script.

**To run the entire pipeline:**
```bash
python3 run.py
```

**To skip specific steps:**
```bash
# Skip validation and cleaning
python3 run.py --skip-validation --skip-cleaning

# Run only the training step
python3 run.py --skip-validation --skip-cleaning --skip-preparation
```

## 4. The Data Pipeline (A-Z)

The pipeline is designed to be a fully automated, reproducible, and auditable system for growing the training dataset.

### Ground Rules
- The `dataset/raw/` directory is **append-only**. Raw files are never manually edited.
- All data transformations are performed by version-controlled scripts.
- Any two contributors with the same git commit will always be able to regenerate the exact same cleaned and processed data.

### Phase 1: URL Discovery
- **Goal:** Maintain a growing list of candidate URLs.
- **Process:** URLs are discovered using Google Search with targeted queries.
- **Output:** A manifest file at `dataset/metadata/urls.jsonl`, where each entry logs the URL, query, and status (`new`, `fetched`, `rejected`).

### Phase 2: Fetching & Extraction
- **Goal:** Get the human-readable text from each URL.
- **Process:** A headless browser renders the page, and a "read-mode" algorithm extracts the primary text content, removing UI noise like ads and navigation.
- **Output:** Raw, extracted text is saved to `dataset/raw/text/<hash>.txt`.

### Phase 3: Deduplication
- **Goal:** Prevent dataset rot and content duplication.
- **Process:** Both exact hash-based and near-duplicate text similarity checks are performed.
- **Output:** Metadata is updated to link duplicate content.

### Phase 4: Cleaning & Normalization
- **Goal:** Convert raw text into a model-friendly format.
- **Process:** Scripts normalize whitespace, standardize punctuation, fix encoding issues, and remove site-specific artifacts.
- **Output:** Cleaned text files are written to `dataset/cleaned/`.

### Phase 5: Training Data Preparation
- **Goal:** Create the final `train.txt` and `val.txt` files.
- **Process:** The cleaned text files are concatenated in a deterministic order and split.
- **Output:** The final processed files are stored in `dataset/processed/`.

## 5. Data Format Specification

### Tokenization
- The project uses **character-level tokenization** for its minimal vocabulary size and robustness to noise.

### Control Tokens
- To provide structural context, the following tokens are used:
  - `<|scene_break|>`: Indicates a major scene change.
  - `<|summary|>`: Prefixes a condensed summary of previous text (for the rolling memory mechanism).
  - `<|end_of_text|>`: Marks the end of a distinct document.
- The `<|...|>` format is used to prevent collisions with naturally occurring text.

## 6. Project Roadmap

This roadmap outlines the planned development phases, focusing on a "constraint-first" philosophy.

### Phase 0: Foundations
- **Goal:** A correct, minimal NanoGPT that trains and generates.
- **Status:** Largely complete.

### Phase 1: Hard Constraint Baseline (2 GB Reality Check)
- **Goal:** Make inference runnable on a 2 GB RAM machine.
- **Tasks:** Reduce model size, enforce smaller context windows (`block_size`), and use memory-saving techniques like `torch.no_grad()`.

### Phase 2: Narrative Quality
- **Goal:** Improve story coherence without increasing context length.
- **Tasks:** Introduce control tokens, tune sampling parameters, and train on longer contiguous samples.

### Phase 3: Rolling Summary Memory (Core Novelty)
- **Goal:** Create a mechanism for "pseudo-long-term memory" under a fixed context window.
- **Tasks:** Implement a sliding window generation process where a summary of the previous window is prepended to the next, prefixed by the `<|summary|>` token.

### Phase 4-8: Optimization, Evaluation, and Deployment
- **Goals:** Further micro-optimizations for memory and compute, building robust evaluation metrics, and packaging the model for easy deployment.

## 7. Resource Constraints & Philosophy
This project is designed from the ground up to work with:
- ~2GB RAM (minimum target)
- Limited disk space
- Free-tier Google Colab environments

This is achieved through a small vocabulary, small context windows, small batch sizes, and checkpoint-based training that can be resumed at any time. The core philosophy is to **do more with less**.
