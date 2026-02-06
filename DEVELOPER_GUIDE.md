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
├── .jules/             # Agent-specific instructions, memory, and Bolt Journal
├── dataset/
│   ├── raw/            # Append-only, immutable raw text files
│   ├── cleaned/        # Script-generated cleaned text
│   ├── processed/      # Tokenized, training-ready data files
│   └── metadata/       # URL manifests, source lists, and other metadata
├── scraper/            # Code for discovering and extracting content
├── scripts/            # Automation scripts for data processing and CLI tools
├── training/           # Model definition, training loop, and configs
├── .gitignore          # Specifies intentionally untracked files
├── CONTRIBUTING.md     # The locked, non-negotiable project intent
├── DEVELOPER_GUIDE.md  # This file
├── README.md           # High-level project overview
└── benchmark.py        # Performance benchmarking utility
```

## 3. The Data Pipeline

The data pipeline is the heart of this project. It's a multi-stage process designed to be fully automated and reproducible.

**Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

### Stage 1: URL Discovery (Search Layer)

-   **Command:** `python3 scripts/cli.py search "<query>" --num-results <number>`
-   **Purpose:** Uses search engines to find potential URLs for scraping.
-   **Output:** Appends new URLs to `dataset/metadata/urls.jsonl` with the status "new".

### Stage 2: Content Fetching & Extraction

-   **Command:** `python3 scripts/cli.py process`
-   **Purpose:** Fetches the HTML for URLs marked "new", extracts the readable text content, and saves it.
-   **Output:**
    -   Raw text files are saved in `dataset/raw/source_<source-id>/`.
    -   The corresponding URL entry in `urls.jsonl` is updated to "fetched".

### Stage 3: Data Validation, Cleaning, and Preparation

1.  **Validate Raw Data:**
    -   **Script:** `scripts/validate_raw.py`
    -   **Purpose:** Performs a quick sanity check on the raw text files (minimum length and printable character ratio).

2.  **Clean Dataset (with Language Guard):**
    -   **Script:** `scripts/clean_dataset.py`
    -   **Purpose:** Sanitizes raw text and applies an Indonesian Language Guard to filter out gambling ads and non-narrative content.
    -   **Optimization Note:** The cleaner uses a blacklist of keywords (e.g., "slot gacor") to purge polluted paragraphs, ensuring a high-quality narrative corpus.

3.  **Prepare Data for Training:**
    -   **Script:** `scripts/prepare_data.py`
    -   **Purpose:** Concatenates all cleaned data into a single corpus with structural tokens (`<|story_start|>`, `<|end_of_text|>`).

## 4. Training the Model

-   **Command:** `python3 training/train.py --config training/configs/small.yaml`
-   **Resuming:** The script automatically detects and loads the latest checkpoint from the `out_dir`.

## 5. Generating Text & Performance Metrics

To generate text and measure model performance:

-   **Command:** `python3 scripts/generate.py --config training/configs/small.yaml --checkpoint_path out/model.pt`
-   **Metrics:** The script automatically reports Peak RAM Usage, Generation Time, and Tokens per Second (TPS).
-   **Optimization Note:** The model uses **Positional Tensor Caching** (via `register_buffer`) to reduce CPU-to-GPU overhead during inference.

## 6. Performance & Science (The Bolt Standard)

All major technical decisions are documented in **[.jules/BOLT_JOURNAL.md](.jules/BOLT_JOURNAL.md)**. Contributors should consult this journal to understand the "why" behind:
-   The pivot from `torch.compile` to manual tensor caching.
-   Learning rate selection and micro-training benchmarks.
-   KV-cache implementation details and impact.

## 7. Indonesian Dataset Specialization

For details on curated Indonesian sources and pollution blacklists, see:
-   **[dataset/metadata/SOURCES.md](dataset/metadata/SOURCES.md)** (Curated targets for adult Indonesian content).

## 8. Contribution Workflow

1.  **Sync with `main`**: Always rebase before starting work.
2.  **Document Proofs**: If you optimize the code, you MUST add a scientific entry to the `BOLT_JOURNAL.md`.
3.  **Isolate Features**: Keep new functionality in separate modules.
4.  **No Manual Curation**: Never edit files in `dataset/raw/`.
