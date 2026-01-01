# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Philosophy

`miniature-memory` is an experiment in doing more with less. It is built on a "constraint-first" philosophy, designed to train a coherent language model on a continuously growing dataset with minimal hardware (~2GB RAM).

The project's core is its deterministic, auditable, and fully automated data pipeline, which handles everything from web scraping and text extraction to data cleaning and preparation for training.

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

This project is not a general-purpose framework. It is a focused, opinionated, and resource-aware system for building a very specific type of language model.
