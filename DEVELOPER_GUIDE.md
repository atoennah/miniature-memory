# Developer Guide

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the data pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

## 1. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+.

### Dependencies

To install all required Python packages, run the following command from the root of the repository:

```bash
pip install -r requirements.txt
```

This will install `torch`, `pyyaml`, `psutil`, and other essential libraries.

## 2. Core Project Structure

Understanding the repository layout is key to contributing effectively.

```
.
├── .jules/             # Agent-specific instructions and memory
├── dataset/
│   ├── raw/            # Append-only, immutable raw text files
│   ├── cleaned/        # Script-generated cleaned text
│   ├── processed/      # Tokenized, training-ready data files
│   └── metadata/       # URL manifests and other metadata
├── scraper/            # Code for discovering and extracting content
├── scripts/            # Automation scripts for data processing and CLI tools
├── training/           # Model definition, training loop, and configs
├── .gitignore          # Specifies intentionally untracked files
├── CONTRIBUTING.md     # The locked, non-negotiable project intent
├── DEVELOPER_GUIDE.md  # This file
├── README.md           # High-level project overview
└── ROADMAP.md          # Long-term project goals
```

## 3. The Data Pipeline

The data pipeline is the heart of this project. It's a multi-stage process designed to be fully automated and reproducible.

### Stage 1: URL Discovery (Search)

-   **Command:** `python3 scripts/cli.py search "<query>" --num-results <number>`
-   **Purpose:** Uses Google Search to find potential URLs for scraping.
-   **Output:** Appends new URLs to `dataset/metadata/urls.jsonl` with the status "new".

### Stage 2: Content Fetching & Extraction

-   **Command:** `python3 scripts/cli.py process`
-   **Purpose:** Fetches the HTML for URLs marked "new", extracts the readable text content, and saves it.
-   **Output:**
    -   Raw text files are saved in `dataset/raw/source_<source-id>/`.
    -   The corresponding URL entry in `urls.jsonl` is updated to "fetched".

### Stage 3: Data Validation, Cleaning, and Preparation

This is a sequence of three commands that must be run in order.

1.  **Validate Raw Data:**
    -   **Command:** `python3 scripts/validate_raw.py`
    -   **Purpose:** Checks raw files for basic integrity.

2.  **Clean Dataset:**
    -   **Command:** `python3 scripts/clean_dataset.py`
    -   **Purpose:** Normalizes whitespace, fixes encoding issues, and removes artifacts from the raw text. Generates the `dataset/cleaned/` directory.

3.  **Prepare Data for Training:**
    -   **Command:** `python3 scripts/prepare_data.py`
    -   **Purpose:** Concatenates the cleaned data, performs character-level tokenization, and creates the final `train.txt` and `val.txt` files.
    -   **Output:** `dataset/processed/train.txt` and `dataset/processed/val.txt`.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

## 4. Training the Model

Once the dataset is processed, you can train the model.

### Local Training

-   **Command:** `python3 training/train.py --config training/configs/small.yaml`
-   **Function:** Starts the training loop using the specified configuration file and the data in `dataset/processed/`.
-   **Checkpoints:** Model checkpoints are saved periodically to the `out/` directory (by default).

### Resuming Training

Training is designed to be resumable. If the script is interrupted, simply run the same command again, and it will automatically load the latest checkpoint from the `out_dir` specified in the config.

## 5. Generating Text

To generate text from a trained model, use the generation script.

-   **Command:** `python3 scripts/generate.py --max_new_tokens <number>`
-   **Function:** Loads the latest checkpoint and generates a sample of the specified length.

## 6. Contribution Workflow

1.  **Sync with `main`:** Before starting any work, ensure your local branch is up-to-date with the latest `main`.
2.  **Isolate Your Changes:** New features should be developed in separate modules or files whenever possible to minimize conflicts.
3.  **Follow the Data Rules:** If your contribution touches the dataset, you must adhere to the append-only and script-driven transformation rules.
4.  **Submit for Review:** All changes are reviewed by the Curator and the BDFL before being merged. Ensure your work is clean, documented, and aligned with the project's intent as described in `CONTRIBUTING.md`.
