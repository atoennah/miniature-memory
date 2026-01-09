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
- **Environment:** CPU / small GPU

The goal is to push NanoGPT as far as possible under minimal resources.

## Project Documentation

This project is managed with a strong focus on clear, centralized documentation to ensure alignment and long-term stability. For detailed information, please refer to the documents in the `.jules/` directory.

-   **`.jules/AGENTS.md`**: Defines the roles, responsibilities, and governance model for all agents and contributors.
-   **`.jules/ROADMAP.md`**: Outlines the complete, unified development plan for the data pipeline and model evolution.

## Quick Start

### 1. Setup

To set up the environment and install all necessary dependencies, run the setup script:
```bash
./setup.sh
```

### 2. Run the Pipeline

The entire data processing and training pipeline is orchestrated by the `run.py` script.

**To run the full pipeline:**
```bash
python3 run.py
```

**To skip specific steps:**
```bash
# Skip data validation and cleaning
python3 run.py --skip-validation --skip-cleaning

# Run only the training step
python3 run.py --skip-validation --skip-cleaning --skip-preparation
```

### 3. Google Colab

For a notebook-driven workflow, open and run the cells in `NanoGPT_Adult_Training.ipynb`.

## Core Principles

-   **Constraint-First:** All work is optimized for tight resource limits (~2GB RAM).
-   **Automation:** The entire data and training pipeline is designed to be fully automated and reproducible.
-   **Incremental Growth:** The dataset grows continuously, and the model is trained incrementally from checkpoints.

No black boxes, no magic, no assumptions.

