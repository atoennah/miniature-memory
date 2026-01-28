# The Data Pipeline

This document provides a detailed, step-by-step walkthrough of the data pipeline. For a high-level overview, project setup, and contribution guidelines, please see the main **[README.md](../README.md)**.

The data pipeline is a multi-stage process designed to be fully automated and reproducible. It transforms raw, scraped web content into a clean, tokenized format ready for training.

**The Golden Rule:** The `dataset/raw/` directory is sacred. Raw data is append-only and must never be manually edited. All transformations are handled by version-controlled scripts to ensure 100% reproducibility.

---

## Stage 1: URL Discovery (Search)

-   **Command:** `python3 scripts/cli.py search "<query>" --num-results <number>`
-   **Purpose:** Uses Google Search to find potential URLs for scraping. This is the first step in expanding the dataset.
-   **Input:** A search query string and the desired number of results.
-   **Output:** Appends new URLs to `dataset/metadata/urls.jsonl` with the status "new". Each entry is a JSON object containing the URL and its current status.

---

## Stage 2: Content Fetching & Extraction

-   **Command:** `python3 scripts/cli.py process`
-   **Purpose:** Fetches the HTML content for all URLs currently marked "new" in the metadata file, extracts the core narrative text, and saves it to the raw dataset directory.
-   **Input:** `dataset/metadata/urls.jsonl`
-   **Output:**
    -   Raw text files are saved in `dataset/raw/source_<source-id>/`. Each file corresponds to a successfully fetched and processed URL.
    -   The corresponding URL entry in `urls.jsonl` is updated from "new" to "fetched".

---

## Stage 3: Data Validation, Cleaning, and Preparation

This is a sequence of three automated, script-driven steps that process the raw data into a final, training-ready format.

### 3.1. Validate Raw Data

-   **Script:** `scripts/validate_raw.py`
-   **Purpose:** Performs a quick sanity check on the raw text files. It ensures files have a minimum length and a high ratio of printable characters, filtering out empty, corrupted, or non-narrative data.
-   **Input:** Files in `dataset/raw/`.
-   **Output:** Moves invalid files to `dataset/rejected/` for later inspection.

### 3.2. Clean Dataset

-   **Script:** `scripts/clean_dataset.py`
-   **Purpose:** Sanitizes the validated raw text. This includes normalizing whitespace, removing non-narrative characters (like social media sharing buttons), and stripping excessive newlines to create a clean, readable corpus.
-   **Input:** Files in `dataset/raw/`.
-   **Output:** Cleaned text files are written to the `dataset/cleaned/` directory.

### 3.3. Prepare Data for Training

-   **Script:** `scripts/prepare_data.py`
-   **Purpose:** This is the final step before training. It concatenates all cleaned data into a single corpus, performs the character-level tokenization as defined in **[DATA_FORMAT.md](../DATA_FORMAT.md)**, and splits the result into training and validation sets.
-   **Input:** Files in `dataset/cleaned/`.
-   **Output:**
    -   `dataset/processed/train.txt`
    -   `dataset/processed/val.txt`

---

## Stage 4: Training the Model

Once the dataset is processed, the model can be trained.

-   **Command:** `python3 -m training.train --config training/configs/small.yaml`
-   **Purpose:** Starts the training loop using the hyperparameters specified in the YAML configuration file. The script will automatically detect and load the latest checkpoint from the output directory to resume training if interrupted.
-   **Checkpoints:** Model checkpoints are saved periodically to the `out/` directory.

---

## Stage 5: Generating Text

-   **Command:** `python3 -m scripts.generate --max_new_tokens <number>`
-   **Purpose:** Loads the latest trained model checkpoint and generates a sample of the specified length. This is used to evaluate the model's performance and narrative capabilities.
