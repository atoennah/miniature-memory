# System Architecture

This document provides a high-level overview of the `miniature-memory` system architecture. Its purpose is to explain the "why" behind the design and how the different components work together to create a fully automated, reproducible, and resource-constrained training pipeline.

## 1. Architectural Principles

The entire system is built on three core principles that inform every design decision.

### Constraint-First Design
The pipeline is engineered to operate in environments with significant resource limitations (e.g., low RAM, no GPU). This principle dictates key choices:
- **Character-Level Tokenization:** Keeps the model's vocabulary and memory footprint to a minimum.
- **Small Model Architecture:** The default model (`small.yaml`) is a NanoGPT-style transformer designed to be trainable on modest hardware.
- **Script-Driven Processing:** Avoids the need for large, memory-intensive data processing libraries.

### Absolute Reproducibility
The data pipeline is deterministic. Given the same set of raw files, the final `train.txt` will be identical every time. This is enforced by:
- **Immutable Raw Data:** The `dataset/raw/` directory is append-only. Raw source material is never modified once added.
- **Scripted Transformations:** All data cleaning and preparation steps are handled by version-controlled Python scripts. There is **zero manual data curation**.

### Unwavering Intent Alignment
The project's purpose is locked to adult entertainment. The architecture must directly serve this goal. This means the system is optimized for processing and modeling narrative, often explicit, text found on the web, without implementing sanitization or content-agnostic features.

## 2. System Components

The architecture is composed of three primary, decoupled components: the Scraper, the Data Pipeline, and the Trainer.

![System Diagram](https://i.imgur.com/9z2r78q.png)

### The Scraper
- **Location:** `scraper/`
- **Purpose:** To discover and fetch raw, unstructured text from the internet.
- **Function:** It uses search queries to find potential URLs, scrapes the HTML content, and extracts the core narrative text. It is the only component that interacts with the outside world.
- **Output:** Raw text files are saved to the `dataset/raw/` directory, and metadata is updated in `dataset/metadata/urls.jsonl`.

### The Data Pipeline
- **Location:** `scripts/`
- **Purpose:** To transform raw, messy text into clean, structured, training-ready data.
- **Function:** This is a series of automated, sequential scripts that perform validation, cleaning (e.g., whitespace normalization), and tokenization. It acts as the reproducible "factory" that builds the final dataset.
- **Output:** The final `train.txt` and `val.txt` files in `dataset/processed/`.

### The Trainer
- **Location:** `training/`
- **Purpose:** To train the language model.
- **Function:** This component contains the `GPT` model definition (`model.py`), the `Trainer` class (`trainer.py`) that manages the training loop, and YAML configuration files (`configs/`). It loads the processed data and produces a trained model.
- **Output:** Model checkpoints saved to the `out/` directory.

## 3. Data Flow

The data flows through the system in a clear, linear progression. Each stage produces an intermediate artifact that serves as the input for the next stage.

1.  **URL Manifest (`dataset/metadata/urls.jsonl`)**
    - The starting point. A list of URLs, each with a status (e.g., "new", "fetched").

2.  **Raw Data (`dataset/raw/`)**
    - The Scraper fetches URLs and saves the extracted, unmodified text here. This directory is treated as an immutable, append-only log.

3.  **Cleaned Data (`dataset/cleaned/`)**
    - The first step of the Data Pipeline processes the raw data into a normalized format, handling issues like extraneous whitespace or non-narrative elements.

4.  **Processed Data (`dataset/processed/`)**
    - The final step of the pipeline tokenizes the cleaned text and concatenates it into two files: `train.txt` and `val.txt`. This is the direct input for the Trainer.

5.  **Model Checkpoints (`out/`)**
    - The final output. The Trainer uses the processed data to train the model and saves its weights periodically to this directory.
