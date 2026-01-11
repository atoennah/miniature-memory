# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models, designed for low-RAM (~2GB) environments.

This project provides a complete, from-scratch pipeline to:

-   Scrape and process text data from the web.
-   Build a growing, version-controlled dataset.
-   Train a small-scale, NanoGPT-style language model.
-   Run and resume training under tight resource constraints.

The core philosophy is **constraint-first development**: every architectural choice is optimized for simplicity, reproducibility, and minimal resource usage.

## Key Documents

-   **[Developer Guide](DEVELOPER_GUIDE.md):** Detailed technical documentation, setup instructions, and pipeline overview. **Start here if you are a contributor.**
-   **[Roadmap](ROADMAP.md):** The long-term development plan and project vision.
-   **[Data Format](DATA_FORMAT.md):** The specification for training data and control tokens.

## Quick Start

1.  **Setup the environment:**
    ```bash
    ./setup.sh
    ```

2.  **Run the full pipeline (data processing and training):**
    ```bash
    python3 run.py
    ```

For detailed instructions on workflow, configuration, and dataset design, please read the [Developer Guide](DEVELOPER_GUIDE.md).

## Project Goals

-   To build a fully transparent, "no black boxes" training pipeline.
-   To explore the limits of small language models under severe hardware constraints.
-   To create a stable, reproducible framework for incremental dataset growth and model training.

This is not a large-model framework. It is an experiment in doing more with less.
A minimal, memory-aware dataset and training pipeline for small GPT-style models.

This project is an experiment in doing more with less. We are building, growing, and training a small-but-mighty text-based language model from scratch, optimized for tight resource constraints (like the free tier of Google Colab).

Our philosophy is "constraint-first." Instead of scaling up, we're scaling smart, focusing on a clean, ever-growing dataset and a fully-automated pipeline.
A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Philosophy

`miniature-memory` is an experiment in doing more with less. It is built on a "constraint-first" philosophy, designed to train a coherent language model on a continuously growing dataset with minimal hardware (~2GB RAM).

The project's core is its deterministic, auditable, and fully automated data pipeline, which handles everything from web scraping and text extraction to data cleaning and preparation for training.

- Scrapes text content from the web.
- Extracts readable text using a browser-style “read mode.”
- Stores raw text in a carefully structured, append-only dataset.
- Cleans, prepares, and tokenizes data for training.
- Incrementally trains a small GPT-style model.
- Saves checkpoints so training can resume anytime.

This project is not a large-model framework. It is intentionally small, explicit, and designed for learning and experimentation.

## Getting Started

Welcome! Whether you're a new developer or a seasoned machine learning engineer, we have a place for you.

- **To understand the project's vision and future,** start with our **[ROADMAP.md](ROADMAP.md)**.
- **For a full technical breakdown, setup instructions, and contribution guidelines,** please see our **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**.

## Our Philosophy

We believe that great things can be built with limited resources. This project is a testament to that idea. By focusing on a strong data foundation and a reproducible, automated workflow, we aim to push a NanoGPT-style model as far as it can go.
**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** For a full breakdown of this non-negotiable mission, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Key Features

-   **Automated Data Pipeline:** From Google Search to a training-ready `train.txt`, the entire data workflow is scripted and reproducible.
-   **Constraint-First Design:** Optimized for low-RAM, CPU-only, and free-tier environments.
-   **Incremental Growth:** The dataset is designed to grow continuously without requiring training restarts.
-   **NanoGPT-from-Scratch:** A minimal, understandable, and hackable implementation of a GPT-style model.

## Getting Started

For contributors, the **[Developer Guide](DEVELOPER_GUIDE.md)** is the single source of truth. It provides a comprehensive walkthrough of the environment setup, data pipeline, training commands, and contribution workflow.

### Quick Commands

-   **Install dependencies:** `pip install -r requirements.txt`
-   **Run the full data pipeline:** See the step-by-step commands in the [Developer Guide](DEVELOPER_GUIDE.md#3-the-data-pipeline).
-   **Start training:** `python3 training/train.py --config training/configs/small.yaml`
-   **Generate text:** `python3 scripts/generate.py --max_new_tokens 500`

## Project Documentation

-   **[Developer Guide](DEVELOPER_GUIDE.md):** The essential guide for all contributors. **Start here.**
-   **[CONTRIBUTING.md](CONTRIBUTING.md):** The locked project intent and contribution rules.
-   **[ROADMAP.md](ROADMAP.md):** The long-term vision and development phases.
-   **[DATA_FORMAT.md](DATA_FORMAT.md):** Technical specification for the training data format.
-   **Agent & Memory Files:** The `.jules/` directory contains internal documentation for the AI agents that help maintain this repository.
```
.
├── dataset/
│   ├── raw/          # raw extracted text (one file per content)
│   ├── cleaned/      # cleaned text (derived)
│   └── processed/    # tokenized / training-ready data
│
├── scraper/
│   ├── search/       # Google search helpers
│   ├── fetch/        # URL fetching & rendering
│   └── extract/      # read-mode text extraction
│
├── scripts/
│   ├── validate_raw.py
│   ├── clean_dataset.py
│   ├── prepare_data.py
│   └── stats.py
│
├── training/
│   ├── model.py
│   ├── train.py
│   └── configs/
│
├── run.py
├── setup.sh
├── NanoGPT_Training.ipynb
└── README.md
```

No black boxes, no magic, no assumptions. Just clean code and a clear process.

Join us!

This project is not a general-purpose framework. It is a focused, opinionated, and resource-aware system for building a very specific type of language model.
