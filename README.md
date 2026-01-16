# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** For a full breakdown of this non-negotiable mission, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Project Philosophy

`miniature-memory` is an experiment in doing more with less. It is built on a "constraint-first" philosophy, designed to train a coherent language model on a continuously growing dataset with minimal hardware (~2GB RAM).

The project's core is its deterministic, auditable, and fully automated data pipeline, which handles everything from web scraping and text extraction to data cleaning and preparation for training.

## Getting Started

For contributors, the **[Developer Guide](DEVELOPER_GUIDE.md)** is the single source of truth. It provides a comprehensive walkthrough of the environment setup, data pipeline, training commands, and contribution workflow.

### Quick Start

1.  **Setup the environment:**
    ```bash
    ./setup.sh
    ```

2.  **Run the full pipeline (data processing and training):**
    ```bash
    python3 run.py
    ```

## Project Documentation

-   **[Developer Guide](DEVELOPER_GUIDE.md):** The essential guide for all contributors. **Start here.**
-   **[CONTRIBUTING.md](CONTRIBUTING.md):** The locked project intent and contribution rules.
-   **[ROADMAP.md](.jules/ROADMAP.md):** The long-term vision and development phases.
-   **[DATA_FORMAT.md](DATA_FORMAT.md):** Technical specification for the training data format.
-   **Agent & Memory Files:** The `.jules/` directory contains internal documentation for the AI agents that help maintain this repository.
