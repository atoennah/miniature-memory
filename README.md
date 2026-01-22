# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Purpose: Adult Entertainment

**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** This is a locked, non-negotiable mission. For a full breakdown of the project's intent and the strict rules governing contributions, see our contributor guide:

-   **[CONTRIBUTING.md](CONTRIBUTING.md)**: The locked project intent and contribution rules. **All contributors must read this.**

## Documentation Hub

This repository is managed with a strong focus on clear, centralized documentation. To understand the project, please use the following resources as your guide.

-   **To understand *how* to set up the project and use the pipeline:**
    -   **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md):** The essential guide for all contributors. Contains setup instructions, a full data pipeline walkthrough, and training commands. **Start here.**

-   **To understand the project's *intent* and *governance*:**
    -   **[CONTRIBUTING.md](CONTRIBUTING.md):** The locked, non-negotiable project intent.
    -   **[.jules/AGENTS.md](.jules/AGENTS.md):** The roles and governance model for all agents and contributors.

-   **To understand the project's *long-term vision* and *history*:**
    -   **[.jules/ROADMAP.md](.jules/ROADMAP.md):** The long-term vision and development phases.
    -   **[.jules/BOLT_JOURNAL.md](.jules/BOLT_JOURNAL.md):** A scientific log of all major architectural decisions and performance benchmarks.

## Quick Start

1.  **Setup the environment:**
    ```bash
    ./setup.sh
    ```

2.  **Run the full pipeline (data processing and training):**
    ```bash
    python3 run.py
    ```

For more detailed commands, including how to run specific steps of the pipeline, please see the [Developer Guide](DEVELOPER_GUIDE.md).
