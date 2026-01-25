# miniature-memory

A minimal, memory-aware, and fully automated pipeline for training small, specialized GPT-style language models under significant resource constraints.

## Project Purpose: Adult Entertainment

**This project is explicitly and exclusively dedicated to building language models for adult entertainment.** This is a locked, non-negotiable mission. For a full breakdown of the project's intent and the strict rules governing contributions, see our contributor guide:

-   **[CONTRIBUTING.md](CONTRIBUTING.md)**: The locked project intent and contribution rules. **All contributors must read this.**

## Documentation Hub

This repository is managed with a strong focus on clear, centralized documentation. To understand the project, please use the following resources as your guide.

-   **To understand *how* to contribute and use the pipeline:**
    -   **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md):** The canonical source of truth for the project's technical workflow. It provides setup instructions and a complete walkthrough of the `run.py` orchestrator. **All developers must start here.**

-   **To understand the project's governance and *why* decisions are made:**
    -   **[.jules/AGENTS.md](.jules/AGENTS.md):** Defines the roles and governance model for all agents and contributors.
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

For a detailed explanation of the pipeline stages and advanced `run.py` commands (e.g., skipping steps), please consult the canonical [Developer Guide](DEVELOPER_GUIDE.md).
