# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Philosophy

This project is built on three core principles:

1.  **Constraint-First:** All development prioritizes functionality within tight resource limits (~2GB RAM).
2.  **Automation:** All data processing is scripted, deterministic, and reproducible. No manual data curation is allowed.
3.  **Intent-Locked:** The project is explicitly and exclusively dedicated to building language models for adult entertainment.

For a full breakdown of the project's intent and the strict rules governing contributions, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## The Data Pipeline in 60 Seconds

The project's heart is an automated pipeline that turns raw web content into clean, tokenized training data.

1.  **Discover:** The `discover` script uses Google Search to find potential story URLs and saves them to a manifest file.
2.  **Scrape:** The `scrape` script fetches the HTML from new URLs and extracts the core narrative text.
3.  **Clean:** The `clean` script normalizes whitespace, removes unwanted characters, and strips non-narrative content.
4.  **Prepare:** The `prepare` script tokenizes the cleaned text and creates the final `train.txt` and `val.txt` files for the model.

This entire process is orchestrated by the `run.py` script, which ensures that the dataset is always reproducible from the raw source files.

## Getting Started

### 1. Setup the Environment

```bash
./setup.sh
```

This script installs all necessary Python dependencies from `requirements.txt`.

### 2. Run the Full Pipeline

```bash
python3 run.py
```

This command will execute the entire data pipeline and then begin training the model.

### 3. Run a Specific Stage

You can also run individual stages of the pipeline. For a full list of commands and options, please refer to the **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**.

## Documentation Hub

-   **To understand *how* to contribute and use the pipeline:**
    -   **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md):** The essential technical manual for all contributors. **Start here.**

-   **To understand the project's governance and *why* decisions are made:**
    -   **[.jules/AGENTS.md](.jules/AGENTS.md):** Defines the roles and governance model for all agents and contributors.
    -   **[.jules/ROADMAP.md](.jules/ROADMAP.md):** The long-term vision and development phases.
    -   **[.jules/BOLT_JOURNAL.md](.jules/BOLT_JOURNAL.md):** A scientific log of all major architectural decisions and performance benchmarks.
