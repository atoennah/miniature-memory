# Developer Guide

Welcome, contributor. This guide is your single source of truth for setting up your environment, understanding the automated pipeline, and contributing to the `miniature-memory` project. Adherence to these guidelines is essential for maintaining the quality and reproducibility of our work.

## 1. Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+.

### Dependencies

To install all required Python packages, run the following command from the root of the repository:

```bash
./setup.sh
```

This will install `torch`, `pyyaml`, `psutil`, and other essential libraries.

## 2. The Automated Pipeline (`run.py`)

The primary workflow for this project is orchestrated by the `run.py` script. This script automates the entire data processing and training pipeline, ensuring consistency and reproducibility.

**The Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

### Standard Workflow

To run the entire pipeline—from validating raw data to cleaning, preparing, and finally training the model—use the following command:

```bash
python3 run.py
```

This single command executes the four core stages of the project in the correct order.

### Pipeline Stages

1.  **Validation:** Performs a sanity check on raw text files in `dataset/raw/` to ensure they are well-formed.
2.  **Cleaning:** Sanitizes the raw text by normalizing whitespace and removing non-narrative artifacts. Writes output to `dataset/cleaned/`.
3.  **Preparation:** Concatenates cleaned data, tokenizes it, and creates the final `train.txt` and `val.txt` files in `dataset/processed/`.
4.  **Training:** Starts the model training process using the prepared dataset.

### Controlling the Pipeline

You can run specific parts of the pipeline by using flags to skip stages. This is useful for development, debugging, or when you only need to retrain the model on already-processed data.

-   `--skip-validation`: Skips the initial validation step.
-   `--skip-cleaning`: Skips the data cleaning step.
-   `--skip-preparation`: Skips the data preparation step.
-   `--skip-training`: Skips the model training step.
-   `--no-sync`: Disables the automatic sync with the Hugging Face Hub (if configured).

**Example:** To run only the training step with a specific configuration, assuming the data is already processed:

```bash
python3 run.py --skip-validation --skip-cleaning --skip-preparation --config training/configs/medium.yaml
```

## 3. Training & Configuration

While `run.py` orchestrates the training process, the model's architecture and hyperparameters are managed via YAML configuration files located in `training/configs/`.

-   **`model`**: Defines the architecture (embedding size, number of heads, layers, etc.).
-   **`training`**: Defines the training parameters (batch size, learning rate, max steps).

The default configuration is `training/configs/small.yaml`. You can specify a different one using the `--config` flag with `run.py`.

## 4. Generating Text

To generate text from the latest trained model, use the `scripts/generate.py` script.

-   **Command:** `python3 scripts/generate.py --max_new_tokens <number>`
-   **Function:** Loads the most recent checkpoint from the `out/` directory and generates a sample of the specified length.

## 5. Advanced Usage & Debugging

For development or debugging purposes, you may need to run pipeline stages manually. The following scripts are called by the `run.py` orchestrator.

-   **Validate Raw Data:**
    -   `python3 scripts/validate_raw.py`
-   **Clean Dataset:**
    -   `python3 scripts/clean_dataset.py`
-   **Prepare Data for Training:**
    -   `python3 scripts/prepare_data.py`
-   **Run Training Manually:**
    -   `python3 training/train.py --config training/configs/small.yaml`

## 6. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Use the Orchestrator:** Base your workflow around `run.py`.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
