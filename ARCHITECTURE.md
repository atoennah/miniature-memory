# System Architecture

This document provides a high-level overview of the `miniature-memory` system architecture. Its purpose is to explain the core design principles, the primary components, and how they interact to form a fully automated data processing and model training pipeline.

## 1. Design Principles

The system is built on a foundation of several key principles that ensure reproducibility, scalability, and alignment with the project's core mission.

-   **Automation First:** Every stage of the pipeline, from data collection to model training, is designed to be executed via scripts. Manual intervention is explicitly forbidden in the data pipeline to ensure 100% reproducibility.
-   **Immutability of Raw Data:** The `dataset/raw/` directory is treated as a sacred, append-only archive. Raw source material is never modified in place. All cleaning and transformation steps are performed by scripts that read from the raw data and write to separate directories (`dataset/cleaned/`, `dataset/processed/`). This guarantees that we can always trace our training data back to its exact origin.
-   **Constraint-First Design:** The entire system is optimized to run in resource-constrained environments (low RAM, no GPU). This philosophy influences architectural choices, such as using a character-level tokenizer to keep the vocabulary small and implementing a simple, from-scratch GPT model.
-   **Explicit Intent Alignment:** The architecture is built to serve a single, non-negotiable purpose: generating adult entertainment text. There are no components or design features for content filtering, safety alignment, or multi-purpose use cases.

## 2. Core Components

The system is composed of three primary components that work in sequence: the **Scraper**, the **Data Pipeline**, and the **Trainer**.

![System Flow Diagram](https://i.imgur.com/8a2z3oH.png)

### Component 1: The Scraper

-   **Location:** `scraper/`, `scripts/cli.py`
-   **Purpose:** To discover and fetch raw, unstructured text content from the web.
-   **Function:**
    1.  **Discovery:** Uses search queries to find potential URLs containing relevant narrative content.
    2.  **Extraction:** Fetches the HTML from target URLs and extracts the core text, stripping away boilerplate like ads, navigation menus, and comments.
    3.  **Storage:** Saves the extracted raw text into the `dataset/raw/` directory, organized by source. It also maintains a metadata manifest (`dataset/metadata/urls.jsonl`) to track the status of each URL.

### Component 2: The Data Pipeline

-   **Location:** `scripts/` (specifically `validate_raw.py`, `clean_dataset.py`, `prepare_data.py`)
-   **Purpose:** To transform the raw, messy text into a clean, structured, and tokenized format suitable for training a language model.
-   **Stages:**
    1.  **Validation (`validate_raw.py`):** Performs a quick sanity check on raw files to filter out corrupted or empty data before processing.
    2.  **Cleaning (`clean_dataset.py`):** Normalizes whitespace, removes non-narrative artifacts, and standardizes the text format. The output is written to `dataset/cleaned/`.
    3.  **Preparation (`prepare_data.py`):** Concatenates all cleaned text into a single large corpus. It then performs character-level tokenization, splitting the corpus into training and validation sets (`train.txt`, `val.txt`) stored in `dataset/processed/`.

### Component 3: The Trainer

-   **Location:** `training/`
-   **Purpose:** To train the NanoGPT-style language model on the processed dataset.
-   **Function:**
    -   **Model Definition (`training/model.py`):** Defines the GPT-2 style model architecture, including the self-attention mechanism, transformer blocks, and configuration management.
    -   **Configuration (`training/configs/*.yaml`):** Manages all hyperparameters for both the model architecture and the training process. This separation of code and configuration allows for rapid experimentation.
    -   **Training Loop (`training/train.py`):** Implements the core training logic. It loads the processed data, initializes the model based on a YAML config, runs the optimization loop (forward pass, backward pass, weight update), and periodically saves model checkpoints to the `out/` directory.
    -   **Inference (`scripts/generate.py`):** Uses a trained model checkpoint to generate new text samples.

## 3. Data Flow

The data flow is a linear, one-way process designed for simplicity and reproducibility.

1.  **URLs In:** The process begins with a list of URLs, which are added to `dataset/metadata/urls.jsonl`.
2.  **Raw Text:** The Scraper fetches content from these URLs, creating immutable files in `dataset/raw/`.
3.  **Cleaned Text:** The Data Pipeline reads from `dataset/raw/` and writes sanitized text to `dataset/cleaned/`.
4.  **Processed Data:** The pipeline continues by tokenizing the cleaned text and creating the final `train.txt` and `val.txt` files in `dataset/processed/`.
5.  **Model Training:** The Trainer consumes `dataset/processed/train.txt` to train the model.
6.  **Model Checkpoints Out:** The training process produces binary model checkpoints (`.pt` files) in the `out/` directory, which represent the final output of the entire pipeline.
