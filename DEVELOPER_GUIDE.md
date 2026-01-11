# Developer Guide: miniature-memory

This guide provides a technical overview of the `miniature-memory` project, intended for developers who want to understand, contribute to, or modify the pipeline.

It is the "cache" for your brain. Read this so you don't have to read the code.

## Guiding Philosophy

The project follows a **constraint-first** philosophy. Every component is designed to operate under tight resource limitations (~2GB RAM, free-tier cloud environments). This principle informs every decision, from the choice of a character-level tokenizer to the design of the data pipeline.

---

## The Data Pipeline

The data pipeline is the heart of this project. It's a series of automated, deterministic steps that transform raw, messy web text into a clean, structured corpus ready for training. The entire pipeline is orchestrated by the `run.py` script.

The pipeline consists of three main stages:

1.  **Validation:** `scripts/validate_raw.py`
2.  **Cleaning:** `scripts/clean_dataset.py`
3.  **Preparation:** `scripts/prepare_data.py`

### Stage 1: Validation (`scripts/validate_raw.py`)

-   **Purpose:** To perform a quick sanity check on the raw text files in `dataset/raw/`. This step acts as a gatekeeper, ensuring that only plausible text files proceed to the next stage.
-   **Inputs:** Raw text files (`.txt`) located in `dataset/raw/`.
-   **Output:** Console output indicating whether each file passed or failed validation. No files are created or modified.
-   **Logic:** A file is considered **valid** if it meets two criteria:
    1.  **Minimum Length:** It must contain at least 50 characters. This filters out empty or near-empty files that are artifacts of the scraping process.
    2.  **Printable Character Ratio:** At least 85% of its characters must be "printable" (i.e., standard letters, numbers, punctuation, and whitespace). This is a highly effective heuristic for rejecting binary files, gibberish, or heavily corrupted text.

### Stage 2: Cleaning (`scripts/clean_dataset.py`)

-   **Purpose:** To sanitize the raw text, removing noise and normalizing its structure. The goal is to produce a clean, consistent version of the dataset while preserving the original narrative content.
-   **Inputs:** Raw text files (`.txt`) from `dataset/raw/`.
-   **Outputs:** Cleaned text files (`.txt`) are written to the `dataset/cleaned/` directory, mirroring the original directory structure.
-   **Logic:** The cleaning process involves several sequential operations:
    1.  **Character Whitelisting:** It removes any character that is not a letter, number, common punctuation mark (`.,?!\'"()-`), or whitespace. This is a strict but effective way to eliminate control characters, emojis, and other non-narrative symbols.
    2.  **Whitespace Normalization:** It collapses multiple spaces or tabs into a single space.
    3.  **Newline Reduction:** It reduces three or more consecutive newlines down to a maximum of two. This helps preserve paragraph breaks while eliminating excessive vertical whitespace.
    4.  **Stripping:** It removes any leading or trailing whitespace from the entire file.

### Stage 3: Preparation (`scripts/prepare_data.py`)

-   **Purpose:** To assemble the final training corpus. This is the last step before the data is fed to the model.
-   **Inputs:** All cleaned text files (`.txt`) from the `dataset/cleaned/` directory.
-   **Output:** A single file named `train.txt` in the `dataset/processed/` directory.
-   **Logic:**
    1.  **File Discovery:** The script first finds all `.txt` files within the `dataset/cleaned/` directory.
    2.  **Deterministic Ordering:** To ensure that the `train.txt` file is always identical for a given set of cleaned files, the script sorts the list of file paths alphabetically. This is a critical and intentional design choice for reproducibility.
    3.  **Concatenation:** It reads each cleaned file in the sorted order and appends its content to `train.txt`. A double newline (`\n\n`) is added after each file's content to act as a clear separator between documents.

---

## The Training Process

The training process is handled by `training/train.py`, a script designed for simplicity and clarity. It implements a standard training loop for a NanoGPT-style model.

### How it Works

1.  **Data Loading:** The script begins by loading the entire `dataset/processed/train.txt` corpus into memory.
2.  **Tokenization:** It creates a simple, on-the-fly **character-level tokenizer**. The vocabulary is dynamically generated from the unique set of characters present in the training data. This aligns with the project's philosophy of having no external dependencies and being robust to varied text styles.
3.  **Model Initialization:** It initializes a `GPT` model using the hyperparameters defined in a YAML configuration file.
4.  **Training Loop:** The script then enters the main training loop, which consists of the following repeating steps:
    -   **Batching:** It randomly samples small chunks of data (`x`) and their corresponding targets (`y`, which is `x` shifted by one character) to create a training batch.
    -   **Forward Pass:** The model computes the loss between its predictions and the actual target characters.
    -   **Backward Pass:** It calculates the gradients and updates the model's weights using the AdamW optimizer.
5.  **Checkpointing:** After the training loop is complete, the script saves the final model weights to a file (`model.pt`) in the configured output directory.

### Configuration (`training/configs/small.yaml`)

All hyperparameters for the model and training loop are managed via YAML configuration files. The default configuration is `training/configs/small.yaml`, which is tuned for a quick, low-resource training run.

Key configuration parameters:

-   **`model`**:
    -   `n_embd`: The size of the token embedding vector.
    -   `n_head`: The number of attention heads in each Transformer block.
    -   `n_layer`: The number of Transformer blocks (layers) in the model.
    -   `block_size`: The context window (or sequence length). This is the maximum number of tokens the model can "see" at once.
    -   `dropout`: The dropout rate used for regularization.
-   **`training`**:
    -   `batch_size`: The number of sequences processed in each training step.
    -   `learning_rate`: The step size for the optimizer.
    -   `max_steps`: The total number of training steps to run.
    -   `eval_interval`: How often (in steps) to print the current training loss.

### How to Run Training

While you can run the `training/train.py` script directly, the recommended way is to use the main orchestrator, `run.py`, which ensures the preceding data pipeline steps have been completed.

To run the full pipeline, including training:
`python3 run.py`

To run a training session with a specific configuration file:
`python3 run.py --config path/to/your/config.yaml`

To run *only* the training step, assuming the data is already prepared:
`python3 run.py --skip-validation --skip-cleaning --skip-preparation`
# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models

miniature-memory is a project focused on building, growing, and training a small text-based language model using a NanoGPT-from-scratch approach, optimized for tight resource constraints (e.g. Google Colab, ~2GB RAM environments).

The core of this project is the dataset pipeline: scraping, extracting, storing, cleaning, preparing, and incrementally training on text data in a fully automated way.

## What This Project Does

- Scrapes text content from the web using Google Search as an entry point
- Extracts readable text using browser-style “read mode”
- Stores raw text as an ever-growing dataset
- Cleans and prepares data for training
- Trains a small GPT-style model incrementally
- Saves checkpoints and artifacts so training can resume anytime
- Designed to run on Linux and Google Colab

This project is not a large-model framework.
It is intentionally small, explicit, and constrained.

## Target Model

- **Architecture:** GPT-style decoder-only transformer
- **Base implementation:** NanoGPT-style (from scratch)
- **Scale:** very small models
- **Training:** incremental, resumable
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
This guide provides a detailed technical overview of the miniature-memory project, covering repository structure, setup, and the data pipeline. For a high-level project summary, see the main [README.md](README.md).

## Repository Structure

The repository is organized into distinct modules for data handling, scripting, and training:

```
.
├── dataset/
│   ├── raw/          # Raw extracted text (one file per content)
│   ├── cleaned/      # Cleaned text (derived)
│   └── processed/    # Tokenized / training-ready data
# Developer Guide

This guide provides a technical overview of the miniature-memory project, including repository structure, setup, and the training workflow.

## Repository Structure

```
.
├── dataset/
│   ├── raw/          # Append-only raw extracted text (one file per content item)
│   ├── cleaned/      # Cleaned, normalized text (derived from raw)
│   ├── processed/    # Tokenized, training-ready data
│   ├── rejected/     # Raw content that failed quality checks
│   └── metadata/     # URL manifests and scraping metadata
│   ├── raw/          # raw extracted text (one file per content)
│   ├── cleaned/      # cleaned text (derived)
│   └── processed/    # tokenized / training-ready data
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
│   └── extract/      # read-mode text extraction
│
├── scripts/
│   ├── validate_raw.py
│   ├── clean_dataset.py
│   ├── prepare_data.py
│   └── stats.py
│
├── training/
│   ├── model.py      # Core GPT model definition
│   ├── train.py      # Training loop
│   └── configs/      # YAML configuration for training runs
│
├── .jules/
│   ├── AGENTS.md     # Definitions of agent personas
│   └── BOLT_JOURNAL.md # Log for experimental findings
│
├── run.py            # Main pipeline orchestrator
│   ├── model.py
│   ├── train.py
│   └── configs/
│
├── run.py
├── setup.sh
├── NanoGPT_Training.ipynb
└── README.md
```

## Setup

To set up the development environment and install all dependencies, run the `setup.sh` script:
```bash
./setup.sh
```
This script will create a Python virtual environment and install all packages listed in `requirements.txt`.

## Training Workflow

The entire data processing and training pipeline is orchestrated by the `run.py` script, which provides a unified interface for all stages.

### Local / Linux

To run the complete pipeline from data validation to model training, execute:
```bash
python3 run.py
```

**To skip specific steps:**
```bash
# Skip validation and cleaning
You can also control the pipeline flow using command-line flags to skip specific steps:

```bash
# Skip validation and cleaning, and run only preparation and training
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
To use a specific training configuration, use the `--config` flag:
```bash
python3 run.py --config training/configs/large.yaml
To set up the environment, run the `setup.sh` script:
```bash
./setup.sh
```
This will install all the necessary dependencies.

## Training Workflow

The entire data processing and training pipeline can be run using the unified `run.py` script.

### Local / Linux

To run the entire pipeline, simply execute:
```bash
python run.py
```

You can also skip specific steps using command-line flags:
```bash
# Skip validation and cleaning
python run.py --skip-validation --skip-cleaning

# Run only the training step
python run.py --skip-validation --skip-cleaning --skip-preparation
```

### Google Colab

- Open `NanoGPT_Training.ipynb`.
- Upload or clone the repository into the Colab environment.
- Run all cells from top to bottom.
- Training artifacts and checkpoints are saved periodically to the Colab instance.

## Dataset Design

The dataset is structured in three stages: raw, cleaned, and processed. This design ensures that all transformations are scripted and reproducible.

### Raw Dataset

-   **Content:** Raw data is plain text, stored exactly as extracted from the source.
-   **Structure:** Each piece of content is stored in its own file.
-   **Immutability:** Raw files are considered append-only and should never be edited manually after being written.

Example filename:
`dataset/raw/source_example/20250112_231455__SRC__abc123__9f3a.txt`

The raw dataset serves as the immutable source of truth for the entire pipeline.

### Cleaned Dataset

-   **Derivation:** Generated automatically from the raw data.
-   **Purpose:** Removes noise such as HTML artifacts, encoding errors, and extra whitespace.
-   **Format:** Remains plain text.

### Processed Dataset

-   **Derivation:** Generated from the cleaned data.
-   **Purpose:** This is the training-ready data.
-   **Transformations:** The data is tokenized (character-level) and split into `train.bin` and `val.bin` files.

All cleaned and processed data can be regenerated from the raw dataset at any time, ensuring reproducibility.

## Scraping Pipeline (High Level)

1.  **Discover:** Use Google Search to find candidate URLs based on a query.
2.  **Fetch & Render:** Download the page content and render it in a headless browser to execute JavaScript.
3.  **Extract:** Use a "read mode" algorithm to extract the main narrative text, ignoring boilerplate.
4.  **Save:** Store the extracted text into `dataset/raw/` with a descriptive filename.
5.  **Log:** Record the source URL and timestamp in the metadata manifest.

The scraping pipeline is designed to be repeatable and incremental, allowing the dataset to grow over time.

## Resource Constraints & Design Philosophy

The project is designed to operate under significant resource constraints:

-   **Target RAM:** ~2GB
-   **Environment:** Free-tier Google Colab, low-end CPUs.

This is achieved through several intentional design choices:

-   **Character-Level Tokenization:** Keeps the model's vocabulary and embedding table extremely small.
-   **Small Context Window:** The `block_size` (e.g., 256) limits the size of the attention matrix.
-   **Small Batch Sizes:** Reduces memory pressure during training.
-   **Checkpoint-Based Training:** Allows training to be stopped and resumed at any time.

The core philosophy is **constraint-first development**: every architectural decision is made to support low-resource environments.

## Incremental Growth

The entire pipeline is designed for continuous, incremental growth:

-   New raw data can be added to the `dataset/raw/` directory at any time.
-   The cleaning and preparation scripts run deterministically on the entire dataset.
-   The training script can resume from the latest checkpoint, incorporating new data without restarting from scratch.

This allows the model to evolve and improve as the dataset expands.
- Open `NanoGPT_Training.ipynb`
- Upload or clone repository
- Run all cells top to bottom
- Training artifacts are saved periodically

## Dataset Design

### Raw Dataset

- Raw data is plain text only
- One file = one content item
- Stored exactly as extracted
- Never edited after being written

Example:
`dataset/raw/source_example/20250112_231455__SRC__abc123__9f3a.txt`

Raw data exists only to be consumed by scripts, not humans.

### Cleaned Dataset

- Generated from raw data
- Removes noise (HTML artifacts, encoding issues)
- Normalizes whitespace
- Still text-only

### Processed Dataset

- Tokenized
- Split into train / validation
- Ready for model training

All cleaned and processed data can be re-generated from raw data at any time.

## Scraping Pipeline (High Level)

1.  Search using Google
2.  Collect candidate URLs
3.  Render page (headless / read-mode)
4.  Extract readable text
5.  Save text into `dataset/raw/`
6.  Log source and timestamp

Scraping is designed to be repeatable and incremental.

## Resource Constraints

Designed to work with:

- ~2GB RAM (minimum target)
- Limited disk
- Free-tier Colab environments

Techniques used:

- small vocab
- small context window
- small batch sizes
- checkpoint-based training

## Incremental Growth

- New raw data can be added at any time
- Cleaning and preparation re-run deterministically
- Training resumes from latest checkpoint
- Dataset grows continuously without restarting from zero

## Current Status

- Dataset pipeline: in progress
- Scraping automation: active development
- Training loop: functional
- Colab support: draft / usable

See `ROADMAP.md` for what’s next.

## Summary

miniature-memory is about doing more with less:

- small model
- small machine
- growing dataset
- full control over every step

No black boxes, no magic, no assumptions.
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
