# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Purpose: Adult Entertainment

**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** This is a locked, non-negotiable mission. For a full breakdown of the project's intent and the strict rules governing contributions, see our contributor guide:

-   **[CONTRIBUTING.md](CONTRIBUTING.md)**: The locked project intent and contribution rules. **All contributors must read this.**

## 1. Environment Setup

This project requires Python 3.12+ and a Linux-based environment. To install all required dependencies, run the following command from the root of the repository:

```bash
pip install -r requirements.txt
```

## 2. The Automated Pipeline (Recommended Workflow)

The entire data processing and training pipeline can be executed with a single command. This is the recommended workflow for most use cases.

```bash
python3 run.py
```

This script will automatically:
1.  Validate the raw data.
2.  Clean the dataset.
3.  Prepare the data for training.
4.  Start the training process using the default configuration.

You can customize the pipeline execution with flags. For example, to run only the data processing steps without starting training:

```bash
python3 run.py --skip-training
```

Run `python3 run.py --help` for a full list of options.

## 3. Manual Workflow (For Development & Debugging)

For more granular control, you can run each stage of the pipeline manually.

### 3.1. Data Acquisition

1.  **Discover URLs:**
    -   **Command:** `python3 scripts/cli.py search "<query>" --num-results <number>`
    -   **Purpose:** Uses Google Search to find potential URLs and appends them to `dataset/metadata/urls.jsonl`.

2.  **Fetch Content:**
    -   **Command:** `python3 scripts/cli.py process`
    -   **Purpose:** Fetches the HTML for new URLs, extracts the core text, and saves it to `dataset/raw/`.

### 3.2. Data Processing

The data pipeline transforms raw text from `dataset/raw/` into a training-ready corpus in `dataset/processed/`.

1.  **Validate Raw Data:**
    -   **Command:** `python3 scripts/validate_raw.py`
    -   **Purpose:** Filters out empty or corrupted files from `dataset/raw/`.

2.  **Clean Dataset:**
    -   **Command:** `python3 scripts/clean_dataset.py`
    -   **Purpose:** Normalizes whitespace and removes non-narrative artifacts, writing the output to `dataset/cleaned/`.

3.  **Prepare Data for Training:**
    -   **Command:** `python3 scripts/prepare_data.py`
    -   **Purpose:** Tokenizes the cleaned data and creates the final `train.txt` and `val.txt` in `dataset/processed/`.

### 3.2. Model Training

-   **Command:** `python3 -m training.train --config training/configs/small.yaml`
-   **Purpose:** Starts the training loop using the specified configuration. Checkpoints are saved to the `out/` directory. The `-m` flag is important for correct module resolution.

### 3.3. Text Generation

-   **Command:** `python3 scripts/generate.py --max_new_tokens 500`
-   **Purpose:** Loads the latest trained model and generates a text sample.

## 4. Data Format

The project uses **character-level tokenization** to maintain a small vocabulary and stay robust against typos and slang. To provide narrative structure, the training data uses several control tokens (e.g., `<|scene_break|>`, `<|end_of_text|>`).

For a detailed explanation of the data format, see **[DATA_FORMAT.md](DATA_FORMAT.md)**.

## 5. Project Structure

-   `.jules/`: Governance documents, agent personas, and the project's long-term memory.
-   `dataset/`: The data, organized into `raw`, `cleaned`, and `processed` stages.
-   `scraper/`: Scripts for discovering and fetching new data.
-   `scripts/`: Automation scripts and command-line tools.
-   `training/`: The core model definition, training loop, and configuration files.

## 6. Governance & Long-Term Vision

This project is managed with a strong focus on clear, centralized documentation and governance.

-   **[.jules/AGENTS.md](.jules/AGENTS.md):** Defines the roles and governance model for all agents and contributors.
-   **[.jules/ROADMAP.md](.jules/ROADMAP.md):** The long-term vision and development phases.
-   **[.jules/BOLT_JOURNAL.md](.jules/BOLT_JOURNAL.md):** A scientific log of all major architectural decisions and performance benchmarks.

## 7. Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work.
2.  **Isolate Your Changes:** Develop new features in separate, well-defined modules to minimize merge conflicts.
3.  **Adhere to Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles.
4.  **Submit for Review:** All changes require review from the Curator and final approval from the BDFL. Ensure your work is clean, documented, and fully aligned with the project's intent as defined in `CONTRIBUTING.md`.
