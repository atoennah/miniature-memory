# Developer Guide: miniature-memory

This guide provides a detailed technical overview of the miniature-memory project, covering repository structure, setup, and the data pipeline. For a high-level project summary, see the main [README.md](README.md).

## Repository Structure

The repository is organized into distinct modules for data handling, scripting, and training:

```
.
├── dataset/
│   ├── raw/          # Raw extracted text (one file per content)
│   ├── cleaned/      # Cleaned text (derived)
│   └── processed/    # Tokenized / training-ready data
│
├── scraper/
│   ├── search/       # Google search helpers
│   ├── fetch/        # URL fetching & rendering
│   └── extract/      # Read-mode text extraction
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

You can also control the pipeline flow using command-line flags to skip specific steps:

```bash
# Skip validation and cleaning, and run only preparation and training
python3 run.py --skip-validation --skip-cleaning

# Run only the training step
python3 run.py --skip-validation --skip-cleaning --skip-preparation
```

To use a specific training configuration, use the `--config` flag:
```bash
python3 run.py --config training/configs/large.yaml
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
