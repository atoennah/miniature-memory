# Developer Guide: miniature-memory

This guide provides a technical overview of the `miniature-memory` project, intended for developers who want to understand, contribute to, or modify the pipeline.

It is the "cache" for your brain. Read this so you don't have to read the code.

## Guiding Philosophy

The project follows a **constraint-first** philosophy. Every component is designed to operate under tight resource limitations (~2GB RAM, free-tier cloud environments). This principle informs every decision, from the choice of a character-level tokenizer to the design of the data pipeline.

---

## The Data Pipeline

The data pipeline is the heart of this project. It's a series of automated, deterministic steps that transform raw, messy web text into a clean, structured corpus ready for training. The entire pipeline is orchestrated by the `run.py` script.

The pipeline consists of three main stages:

1.  **Validation:** `scripts/validate_raw.py`
2.  **Cleaning:** `scripts/clean_dataset.py`
3.  **Preparation:** `scripts/prepare_data.py`

### Stage 1: Validation (`scripts/validate_raw.py`)

-   **Purpose:** To perform a quick sanity check on the raw text files in `dataset/raw/`. This step acts as a gatekeeper, ensuring that only plausible text files proceed to the next stage.
-   **Inputs:** Raw text files (`.txt`) located in `dataset/raw/`.
-   **Output:** Console output indicating whether each file passed or failed validation. No files are created or modified.
-   **Logic:** A file is considered **valid** if it meets two criteria:
    1.  **Minimum Length:** It must contain at least 50 characters. This filters out empty or near-empty files that are artifacts of the scraping process.
    2.  **Printable Character Ratio:** At least 85% of its characters must be "printable" (i.e., standard letters, numbers, punctuation, and whitespace). This is a highly effective heuristic for rejecting binary files, gibberish, or heavily corrupted text.

### Stage 2: Cleaning (`scripts/clean_dataset.py`)

-   **Purpose:** To sanitize the raw text, removing noise and normalizing its structure. The goal is to produce a clean, consistent version of the dataset while preserving the original narrative content.
-   **Inputs:** Raw text files (`.txt`) from `dataset/raw/`.
-   **Outputs:** Cleaned text files (`.txt`) are written to the `dataset/cleaned/` directory, mirroring the original directory structure.
-   **Logic:** The cleaning process involves several sequential operations:
    1.  **Character Whitelisting:** It removes any character that is not a letter, number, common punctuation mark (`.,?!\'"()-`), or whitespace. This is a strict but effective way to eliminate control characters, emojis, and other non-narrative symbols.
    2.  **Whitespace Normalization:** It collapses multiple spaces or tabs into a single space.
    3.  **Newline Reduction:** It reduces three or more consecutive newlines down to a maximum of two. This helps preserve paragraph breaks while eliminating excessive vertical whitespace.
    4.  **Stripping:** It removes any leading or trailing whitespace from the entire file.

### Stage 3: Preparation (`scripts/prepare_data.py`)

-   **Purpose:** To assemble the final training corpus. This is the last step before the data is fed to the model.
-   **Inputs:** All cleaned text files (`.txt`) from the `dataset/cleaned/` directory.
-   **Output:** A single file named `train.txt` in the `dataset/processed/` directory.
-   **Logic:**
    1.  **File Discovery:** The script first finds all `.txt` files within the `dataset/cleaned/` directory.
    2.  **Deterministic Ordering:** To ensure that the `train.txt` file is always identical for a given set of cleaned files, the script sorts the list of file paths alphabetically. This is a critical and intentional design choice for reproducibility.
    3.  **Concatenation:** It reads each cleaned file in the sorted order and appends its content to `train.txt`. A double newline (`\n\n`) is added after each file's content to act as a clear separator between documents.

---

## The Training Process

The training process is handled by `training/train.py`, a script designed for simplicity and clarity. It implements a standard training loop for a NanoGPT-style model.

### How it Works

1.  **Data Loading:** The script begins by loading the entire `dataset/processed/train.txt` corpus into memory.
2.  **Tokenization:** It creates a simple, on-the-fly **character-level tokenizer**. The vocabulary is dynamically generated from the unique set of characters present in the training data. This aligns with the project's philosophy of having no external dependencies and being robust to varied text styles.
3.  **Model Initialization:** It initializes a `GPT` model using the hyperparameters defined in a YAML configuration file.
4.  **Training Loop:** The script then enters the main training loop, which consists of the following repeating steps:
    -   **Batching:** It randomly samples small chunks of data (`x`) and their corresponding targets (`y`, which is `x` shifted by one character) to create a training batch.
    -   **Forward Pass:** The model computes the loss between its predictions and the actual target characters.
    -   **Backward Pass:** It calculates the gradients and updates the model's weights using the AdamW optimizer.
5.  **Checkpointing:** After the training loop is complete, the script saves the final model weights to a file (`model.pt`) in the configured output directory.

### Configuration (`training/configs/small.yaml`)

All hyperparameters for the model and training loop are managed via YAML configuration files. The default configuration is `training/configs/small.yaml`, which is tuned for a quick, low-resource training run.

Key configuration parameters:

-   **`model`**:
    -   `n_embd`: The size of the token embedding vector.
    -   `n_head`: The number of attention heads in each Transformer block.
    -   `n_layer`: The number of Transformer blocks (layers) in the model.
    -   `block_size`: The context window (or sequence length). This is the maximum number of tokens the model can "see" at once.
    -   `dropout`: The dropout rate used for regularization.
-   **`training`**:
    -   `batch_size`: The number of sequences processed in each training step.
    -   `learning_rate`: The step size for the optimizer.
    -   `max_steps`: The total number of training steps to run.
    -   `eval_interval`: How often (in steps) to print the current training loss.

### How to Run Training

While you can run the `training/train.py` script directly, the recommended way is to use the main orchestrator, `run.py`, which ensures the preceding data pipeline steps have been completed.

To run the full pipeline, including training:
`python3 run.py`

To run a training session with a specific configuration file:
`python3 run.py --config path/to/your/config.yaml`

To run *only* the training step, assuming the data is already prepared:
`python3 run.py --skip-validation --skip-cleaning --skip-preparation`
