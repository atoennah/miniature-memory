# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Mission: Adult Entertainment Language Models

This project is explicitly and exclusively dedicated to building language models for adult entertainment. This is a locked, non-negotiable mission. All contributions must align with this intent. For a full breakdown of the project's philosophy and the strict rules governing contributions, see **[INTENT.md](INTENT.md)**.

## Project Goals

-   **Constraint-First:** Operate within tight resource limits (~2GB RAM).
-   **Automation:** All data processing is scripted and 100% reproducible.
-   **Intent-Locked:** Exclusively for adult entertainment.

## Environment Setup

This project is designed for a Linux-based environment and requires Python 3.12+.

To install all required Python packages, run the setup script:
```bash
./setup.sh
```
This will install `torch`, `pyyaml`, `psutil`, and other essential libraries from `requirements.txt`.

## Quick Start

To run the full data processing and training pipeline:
```bash
python3 run.py
```

## Project Structure
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
├── README.md           # This file: your starting point
├── INTENT.md           # The locked, non-negotiable project intent
├── PIPELINE.md         # A deep dive into the data pipeline
├── CONTRIBUTING.md     # How to contribute code
└── DATA_FORMAT.md      # Specification for the training data format
```

## Key Architectural Decisions

-   **Character-Level Tokenization:** We use character-level tokenization to maintain a minimal vocabulary and ensure robustness against the typos and stylistic variations common in web-scraped data. This is a deliberate trade-off that prioritizes simplicity and memory efficiency. See **[DATA_FORMAT.md](DATA_FORMAT.md)** for details.
-   **Reproducible Pipeline:** The data pipeline is designed to be 100% reproducible. The `dataset/raw/` directory is append-only, and all transformations are handled by version-controlled scripts. This ensures that the same commit always produces the same dataset. See **[PIPELINE.md](PIPELINE.md)** for a full walkthrough.

## The `.jules` Directory

The `.jules/` directory contains configuration and memory files for **Jules**, the AI software engineer that helps maintain this repository. It defines agent personas, project roadmaps, and stores the "Bolt Journal"—a log of key architectural decisions and benchmark results.

## Documentation Hub

-   **[README.md](README.md):** (This file) Mission, setup, and high-level overview. **Start here.**
-   **[INTENT.md](INTENT.md):** The core philosophy and non-negotiable rules of the project.
-   **[PIPELINE.md](PIPELINE.md):** A detailed technical walkthrough of the data pipeline.
-   **[CONTRIBUTING.md](CONTRIBUTING.md):** Practical guidelines for submitting code.
-   **[DATA_FORMAT.md](DATA_FORMAT.md):** The technical specification for the training data.
