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
│   ├── AGENTS.md       # Governance model and collaboration rules
│   ├── BOLT_JOURNAL.md # Technical decisions and benchmarks
│   └── ROADMAP.md      # Long-term project goals
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
├── DATA_FORMAT.md      # Training data specifications
├── DEVELOPER_GUIDE.md  # This file
└── README.md           # High-level project overview
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

This is a sequence of three scripted steps that process the raw data into a training-ready format.

1.  **Validate Raw Data:**
    -   **Script:** `scripts/validate_raw.py`
    -   **Purpose:** Performs a quick sanity check on the raw text files. It ensures files have a minimum length and a high ratio of printable characters, filtering out empty or corrupted data.

2.  **Clean Dataset:**
    -   **Script:** `scripts/clean_dataset.py`
    -   **Purpose:** Sanitizes the raw text by normalizing whitespace, removing non-narrative characters, and stripping extra newlines.
    -   **Output:** Cleaned text files are written to the `dataset/cleaned/` directory.

3.  **Prepare Data for Training:**
    -   **Script:** `scripts/prepare_data.py`
    -   **Purpose:** Concatenates all cleaned data into a single corpus, performs character-level tokenization, and creates the final `train.txt` and `val.txt` files.
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

## 6. Contribution Workflow

For a detailed breakdown of our agent governance model and collaboration rules, see **[.jules/AGENTS.md](.jules/AGENTS.md)**.

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
