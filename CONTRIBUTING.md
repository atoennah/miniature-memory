# How to Contribute

Thank you for your interest in contributing to the `miniature-memory` project. This document outlines the practical steps and guidelines for submitting your work.

## Core Philosophy

Before you contribute any code, you **must** read and agree to the project's locked, non-negotiable mission statement. This document governs all contributions and is the final authority on the project's direction.

-   **Read the Project Mission:** **[INTENT.md](INTENT.md)**

By submitting a pull request, you are explicitly agreeing to the terms and principles outlined in `INTENT.md`.

## Contribution Workflow

1.  **Sync with `main`:** Always ensure your local branch is up-to-date with the latest `main` before starting work to minimize merge conflicts.
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **Create a Feature Branch:** Create a new branch for your changes.
    ```bash
    git checkout -b your-feature-name
    ```

3.  **Isolate Your Changes:** Develop new features in separate, well-defined modules whenever possible. This makes your code easier to review and reduces the likelihood of conflicts.

4.  **Follow the Data Rules:** All changes that affect the dataset must follow the append-only and script-driven transformation principles outlined in the **[PIPELINE.md](PIPELINE.md)**. **No manual data editing.**

5.  **Run Verification:** Before submitting, ensure the project's benchmark script runs without errors.
    ```bash
    python3 benchmark.py
    ```

6.  **Submit a Pull Request:** Push your branch to the repository and open a pull request against `main`. Provide a clear description of the changes you have made.

## Review Process

All submissions are reviewed by the **Curator Agent** for technical correctness and pipeline integrity, and by the **BDFL Agent** for final approval on intent alignment.

A contribution may be rejected if it does not align with the project's mission as defined in **[INTENT.md](INTENT.md)**, even if it is technically correct.
