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
