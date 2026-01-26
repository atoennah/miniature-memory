# Developer Guide

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the data pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

## 1. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+. To install all required Python packages, run the following command from the root of the repository:

```bash
./setup.sh
```

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

## 3. The Data Pipeline: A Technical Deep Dive

The data pipeline is a multi-stage process designed to be fully automated and reproducible.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

### Stage 1: URL Discovery

-   **Command:** `python3 -m scripts.discover --query "<query>" --num-results <number>`
-   **Purpose:** Uses Google Search to find potential URLs for scraping.
-   **Output:** Appends new URLs to `dataset/metadata/urls.jsonl` with the status "new".

### Stage 2: Content Scraping

-   **Command:** `python3 -m scripts.scrape`
-   **Purpose:** Fetches the HTML for URLs marked "new", extracts the readable text content, and saves it.
-   **Output:**
    -   Raw text files are saved in `dataset/raw/`.
    -   The corresponding URL entry in `urls.jsonl` is updated to "fetched".

### Stage 3: Data Cleaning

-   **Command:** `python3 -m scripts.clean`
-   **Purpose:** Sanitizes the raw text by normalizing whitespace, removing non-narrative characters, and stripping extra newlines.
-   **Output:** Cleaned text files are written to the `dataset/cleaned/` directory.

### Stage 4: Data Preparation

-   **Command:** `python3 -m scripts.prepare`
-   **Purpose:** Concatenates all cleaned data into a single corpus, performs character-level tokenization, and creates the final `train.txt` and `val.txt` files.
-   **Output:** `dataset/processed/train.txt` and `dataset/processed/val.txt`.

### Full Pipeline Orchestration

-   **Command:** `python3 run.py`
-   **Purpose:** Executes the entire data pipeline (discover, scrape, clean, prepare) in the correct order.

## 4. Model Training

Once the dataset is processed, you can train the model using the `training/train.py` script.

### Configuration

All hyperparameters for the model and training loop are managed via YAML configuration files in `training/configs/`.

-   **`model`**: Defines the architecture (embedding size, number of heads, layers, etc.).
-   **`training`**: Defines the training parameters (batch size, learning rate, max steps).

### Running a Training Job

-   **Command:** `python3 -m training.train --config training/configs/small.yaml`
-   **Function:** Starts the training loop using the specified configuration. Checkpoints are saved periodically to the `out/` directory.

### Resuming Training

The script will automatically detect and load the latest checkpoint from the `out_dir` specified in the config.

## 5. Text Generation

To generate text from a trained model, use the `scripts/generate.py` script.

-   **Command:** `python3 -m scripts.generate --max_new_tokens <number>`
-   **Function:** Loads the latest checkpoint and generates a sample of the specified length.

## 6. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
