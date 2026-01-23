# Developer Guide

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the data pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

## 1. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+.

To install all required Python packages and prepare your environment, run the following command from the root of the repository:

```bash
./setup.sh
```

This script will install `torch`, `pyyaml`, `psutil`, and other essential libraries from `requirements.txt`.

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
└── run.py              # The main orchestrator for the entire pipeline
```

## 3. The Unified Pipeline (`run.py`)

The heart of this project is the `run.py` script, which automates the entire data and training workflow. It is designed to be the single entry point for all standard operations, from data processing to model training.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts orchestrated by `run.py` to ensure 100% reproducibility.

### Running the Full Pipeline

To execute the entire pipeline—from validating raw data, cleaning it, preparing it for the model, and finally, running the training—use the following command:

```bash
python3 run.py
```

This command will automatically:
1.  **Validate** the raw data in `dataset/raw/`.
2.  **Clean** the validated data and write it to `dataset/cleaned/`.
3.  **Prepare** the cleaned data by tokenizing it into `dataset/processed/`.
4.  **Train** the model using the default configuration (`training/configs/small.yaml`).

### Controlling the Pipeline with Flags

You can run specific parts of the pipeline by using flags to skip the stages you don't need.

-   **`--skip-validation`**: Skips the initial raw data validation.
-   **`--skip-cleaning`**: Skips the data cleaning step.
-   **`--skip-preparation`**: Skips the final tokenization and data preparation.
-   **`--skip-training`**: Skips the model training step.

**Example:** To run only the data processing steps without starting the training:
```bash
python3 run.py --skip-training
```

### Specifying a Training Configuration

To use a different training configuration file, use the `--config` flag:

```bash
python3 run.py --config training/configs/benchmark.yaml
```

## 4. Generating Text

To generate text from the latest trained model, use the `scripts/generate.py` script. This script will automatically load the most recent checkpoint from the `out/` directory.

-   **Command:** `python3 scripts/generate.py --max_new_tokens <number>`
-   **Function:** Generates a sample of the specified length.

## 5. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
