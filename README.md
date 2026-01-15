# miniature-memory

A minimal, memory-aware dataset and training pipeline for small GPT-style models.

## Project Purpose

This project exists to build, study, and deploy language models for **adult entertainment**. The entire pipeline, from data scraping to model training, is designed for this specific purpose.

The project's intent is locked and non-negotiable. For a full breakdown of this mission and the rules for contributing, please read **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Quick Start

1.  **Setup the environment:**
    ```bash
    ./setup.sh
    ```

2.  **Run the full pipeline (data processing and training):**
    ```bash
    python3 run.py
    ```

For more advanced usage and command-line flags, see the "Training Workflow" section of the [Developer Guide](DEVELOPER_GUIDE.md).

## Project Documentation

This README provides a high-level overview. For a deeper understanding of the project, please consult the following documents:

-   **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md):** The essential technical guide for all contributors. It covers repository structure, setup, and a detailed walkthrough of the data and training pipelines. **Start here if you are a developer.**
-   **[CONTRIBUTING.md](CONTRIBUTING.md):** Defines the project's locked intent, contribution rules, and governance model. **Required reading for all contributors.**
-   **[DATA_FORMAT.md](DATA_FORMAT.md):** The technical specification for the training data format and control tokens.
-   **[.jules/AGENTS.md](./.jules/AGENTS.md):** Defines the roles and responsibilities of the AI agents that help maintain this repository.
-   **[.jules/BOLT_JOURNAL.md](./.jules/BOLT_JOURNAL.md):** A scientific log of all major architectural decisions and performance optimizations.

## Core Principles

-   **Constraint-First:** All work is optimized for tight resource limits (~2GB RAM).
-   **Automation:** The entire data and training pipeline is designed to be fully automated and reproducible.
-   **Incremental Growth:** The dataset grows continuously, and the model is trained incrementally from checkpoints.

## Repository Structure

```
.
├── dataset/
│   ├── raw/          # Append-only raw extracted text
│   ├── cleaned/      # Script-generated cleaned text
│   └── processed/    # Tokenized, training-ready data
│
├── scripts/          # Automation scripts for data processing
├── training/         # Model definition, training loop, and configs
│
├── .jules/           # Agent instructions and project memory
│
├── run.py            # Main pipeline orchestrator script
└── setup.sh          # Environment setup script
```
