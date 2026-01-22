# Developer Guide

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the data pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

## 1. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+. To install all required dependencies, simply run the setup script from the root of the repository:

```bash
./setup.sh
```

This will install `torch`, `pyyaml`, and all other essential libraries from `requirements.txt`.

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
├── run.py              # The main pipeline orchestrator script
└── setup.sh            # The environment setup script
```

## 3. The Automated Pipeline (`run.py`)

The primary workflow for this project is the automated pipeline orchestrated by `run.py`. This single script handles the entire data processing and training sequence, ensuring reproducibility and ease of use.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. The `run.py` script and its underlying modules handle all transformations.

### Running the Full Pipeline

To execute the entire pipeline—from data validation to model training—run the following command:

```bash
python3 run.py
```

This will sequentially:
1.  Validate the raw data in `dataset/raw/`.
2.  Clean the validated data and write it to `dataset/cleaned/`.
3.  Prepare the cleaned data into `dataset/processed/train.txt`.
4.  Start the training process using the default configuration.

### Controlling the Pipeline

You can run specific parts of the pipeline using command-line flags.

-   **Run data preparation only (skip training):**
    ```bash
    python3 run.py --skip-training
    ```

-   **Run training only (assuming data is already processed):**
    ```bash
    python3 run.py --skip-validation --skip-cleaning --skip-preparation
    ```

-   **Disable Hugging Face Hub Sync:**
    The `run.py` script can synchronize with a remote repository on the Hugging Face Hub. To disable this feature for local experiments, use the `--no-sync` flag:
    ```bash
    python3 run.py --no-sync
    ```

## 4. Generating Text

To generate text from a trained model, use the `scripts/generate.py` script. You will need to provide the path to the training configuration and the specific model checkpoint you wish to use.

-   **Example Command:**
    ```bash
    python3 scripts/generate.py --config training/configs/small.yaml --checkpoint_path out/ckpt.pt --max_new_tokens 100
    ```
-   **Note:** Replace `out/ckpt.pt` with the actual path to your trained model checkpoint. Checkpoints are saved in the `out/` directory by default.

## 5. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
