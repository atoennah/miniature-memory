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

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

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

This stage utilizes a modular `processing/` package to transform raw data into a high-quality, training-ready corpus.

1.  **Validate Raw Data:**
    -   **Script:** `scripts/validate_raw.py`
    -   **Purpose:** Performs a sanity check on raw text files, ensuring minimum length and high printable-character ratios. It acts as a "Language Guard" by checking for common Indonesian stop-words.

2.  **Clean Dataset:**
    -   **Script:** `scripts/clean_dataset.py`
    -   **Components:**
        -   `TextNormalizer`: Handles whitespace, Unicode normalization, and punctuation cleaning.
        -   `QualityFilter`: Removes noise based on keywords (e.g., gambling ads, promotional boilerplate) and language registers.
        -   `Segmenter`: Inserts structural markers like `<|story_start|>` and `<|end_of_text|>`.
    -   **Output:** Cleaned text files in `dataset/cleaned/`.

3.  **Prepare Data for Training:**
    -   **Script:** `scripts/prepare_data.py`
    -   **Purpose:** Concatenates all cleaned data into a single corpus and creates the final tokenized `train.txt` and `val.txt` files.
    -   **Output:** `dataset/processed/train.txt` and `dataset/processed/val.txt`.

## 4. Training the Model

Once the dataset is processed, you can train the model using the `training/train.py` script, which is configured via YAML files.

### Configuration (`training/configs/small.yaml`)

All hyperparameters for the model and training loop are managed via YAML configuration files.

-   **`model`**: Defines the architecture (embedding size, number of heads, layers, etc.).
-   **`training`**: Defines the training parameters (batch size, learning rate, max steps).

### Running Training

-   **Command:** `python3 training/train.py --config training/configs/small.yaml`
-   **Function:** Starts the training loop using the specified configuration.
-   **Checkpoints:** Model checkpoints are saved periodically to the `out/` directory (by default), allowing for resumable training.

### Resuming Training

If training is interrupted, simply run the same command again. The script will automatically detect and load the latest checkpoint from the `out_dir` specified in the config.

## 5. Generating Text

To generate text from a trained model, use the `scripts/generate.py` script.

-   **Command:** `python3 scripts/generate.py --max_new_tokens <number>`
-   **Function:** Loads the latest checkpoint and generates a sample of the specified length.

## 6. Technical Insights & Architectural Optimizations

This section documents the "why" behind key architectural decisions, translated from the scientific findings in the [Bolt Journal](.jules/BOLT_JOURNAL.md).

### KV-Caching for Inference
We use a stateful Key-Value (KV) cache in `training/model.py`. This avoids redundant computation of self-attention for past tokens, resulting in a **~4.4x speedup** in generation throughput (from ~44 tok/s to ~197 tok/s on CPU).

### Tied Weights & Optimizer Robustness
To minimize parameter count in memory-constrained environments, we tie the weights of the token embeddings and the language modeling head. The `Trainer` in `training/trainer.py` is designed to handle this by robustly filtering parameters during optimizer construction.

### Pedagogical Code Philosophy
Following our "Conceptual Injection" initiative, core modules like `training/model.py` are designed to be educational. They include detailed mathematical derivations and narrated tensor transformations to reduce cognitive overhead for new developers.

### Indonesian Data Strategy
Our data acquisition focuses on high-volume, modern language registers.
- **Primary Source:** Wattpad Indonesia (using tags like `#dewasa`, `#21plus`, `#gairah`).
- **Pollution Control:** We maintain a strict blacklist for educational or news sites (e.g., `kompas.id`, `halodoc.com`) that may trigger false positives for narrative content.

## 7. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
