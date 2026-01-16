# Developer Guide: miniature-memory

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the data pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

For a high-level project summary, see the main [README.md](README.md).

## 1. Guiding Philosophy & Purpose

### Non-Negotiable Intent
This project exists explicitly and exclusively to build, study, and deploy language models for **adult entertainment**. The training datasets, model architecture, and outputs are intended for adult-oriented text generation. This is a locked project intent. For a full breakdown, you must read [CONTRIBUTING.md](CONTRIBUTING.md).

### Constraint-First Design
The project follows a **constraint-first** philosophy. Every component is designed to operate under tight resource limitations (~2GB RAM, free-tier cloud environments). This principle informs every decision, from the choice of a character-level tokenizer to the design of the data pipeline.

## 2. Repository Structure

The repository is organized to maintain a strict separation of concerns between data, scraping, processing, and training code.

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

## 3. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+.

To install all required Python packages, run the following command from the root of the repository:

```bash
./setup.sh
```

This will create a Python virtual environment and install all packages listed in `requirements.txt`.

## 4. The Data Pipeline (A-Z)

The data pipeline is the heart of this project. It is a series of automated, deterministic steps that transform raw, messy web text into a clean, structured corpus ready for training. The entire pipeline is orchestrated by the `run.py` script.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

The pipeline consists of three main stages:

### Stage 1: Validation (`scripts/validate_raw.py`)

-   **Purpose:** To perform a quick sanity check on the raw text files in `dataset/raw/`. This step acts as a gatekeeper, ensuring that only plausible text files proceed to the next stage.
-   **Inputs:** Raw text files (`.txt`) located in `dataset/raw/`.
-   **Output:** Console output indicating whether each file passed or failed validation. No files are created or modified.
-   **Logic:** A file is considered **valid** if it meets two criteria:
    1.  **Minimum Length:** It must contain at least 50 characters.
    2.  **Printable Character Ratio:** At least 85% of its characters must be "printable".

### Stage 2: Cleaning (`scripts/clean_dataset.py`)

-   **Purpose:** To sanitize the raw text, removing noise and normalizing its structure.
-   **Inputs:** Raw text files (`.txt`) from `dataset/raw/`.
-   **Outputs:** Cleaned text files (`.txt`) are written to the `dataset/cleaned/` directory.
-   **Logic:**
    1.  **Character Whitelisting:** Removes any character that is not a letter, number, common punctuation mark, or whitespace.
    2.  **Whitespace Normalization:** Collapses multiple spaces or tabs into a single space.
    3.  **Newline Reduction:** Reduces three or more consecutive newlines down to a maximum of two.
    4.  **Stripping:** Removes any leading or trailing whitespace.

### Stage 3: Preparation (`scripts/prepare_data.py`)

-   **Purpose:** To assemble the final training corpus.
-   **Inputs:** All cleaned text files (`.txt`) from the `dataset/cleaned/` directory.
-   **Output:** A single file named `train.txt` in the `dataset/processed/` directory.
-   **Logic:**
    1.  **File Discovery:** Finds all `.txt` files in `dataset/cleaned/`.
    2.  **Deterministic Ordering:** Sorts the list of file paths alphabetically for reproducibility.
    3.  **Concatenation:** Appends the content of each cleaned file to `train.txt`, separated by a double newline (`\n\n`).

## 5. The Training Process

The training process is handled by `training/train.py`, designed for simplicity and clarity.

### How it Works

1.  **Data Loading:** Loads the entire `dataset/processed/train.txt` corpus into memory.
2.  **Tokenization:** Creates a simple, on-the-fly **character-level tokenizer**. The vocabulary is dynamically generated from the unique set of characters present in the training data.
3.  **Model Initialization:** Initializes a `GPT` model using the hyperparameters defined in a YAML configuration file.
4.  **Training Loop:** Enters the main training loop, which consists of batching, forward pass, and backward pass.
5.  **Checkpointing:** Saves the final model weights to the configured output directory.

### How to Run the Pipeline

The `run.py` script is the main orchestrator for the entire pipeline.

-   **Run the full pipeline (data processing and training):**
    ```bash
    python3 run.py
    ```
-   **Skip specific steps:**
    ```bash
    # Run only the training step
    python3 run.py --skip-validation --skip-cleaning --skip-preparation
    ```
-   **Use a specific training configuration:**
    ```bash
    python3 run.py --config training/configs/large.yaml
    ```

## 6. Contribution Workflow

1.  **Sync with `main`:** Before starting any work, ensure your local branch is up-to-date with the latest `main`.
2.  **Isolate Your Changes:** New features should be developed in separate modules or files whenever possible to minimize conflicts.
3.  **Follow the Data Rules:** If your contribution touches the dataset, you must adhere to the append-only and script-driven transformation rules.
4.  **Submit for Review:** All changes are reviewed by the Curator and the BDFL before being merged. Ensure your work is clean, documented, and aligned with the project's intent as described in `CONTRIBUTING.md`.
