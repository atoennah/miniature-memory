# Project Roadmap

This document outlines the unified development plan for the `miniature-memory` project, covering both the data pipeline and the model evolution. It serves as the single source of truth for the project's trajectory.

## Guiding Principles

-   **Constraint-First:** All development prioritizes functionality within tight resource limits (~2GB RAM).
-   **Automation:** All data processing is scripted, deterministic, and reproducible.
-   **Intent-Locked:** The project remains focused on its adult-entertainment purpose.

---

## Part 1: Data Pipeline - The Foundation (Completed)

This phase focused on building a robust, automated pipeline to continuously discover, process, and prepare high-quality training data.

### Phase 1.0: Foundations & Workflow (Done)
-   Established canonical directory structure and ground rules (append-only raw data, scripted transformations).

### Phase 1.1: URL Discovery (Search Layer) (Done)
-   Implemented heuristic crawling and manifest management via `urls.jsonl`.

### Phase 1.2: Fetching & Extraction (Done)
-   Implemented headless browser fetching and Trafilatura-based content extraction.

### Phase 1.3: Deduplication & Quality Control (Done)
-   Implemented language guards and printable character filters.

### Phase 1.4: Cleaning & Normalization (Done)
-   Implemented automated cleaning scripts for narrative normalization.

### Phase 1.5: Preparation & Automation (Done)
-   Completed the end-to-end `run.py` pipeline and tokenization scripts.

---

## Part 2: Model Evolution - From NanoGPT to Memory-Aware (In Progress)

This phase focuses on developing a small, efficient GPT-style model capable of generating coherent narratives under extreme memory constraints.

### Phase 2.0: Foundational Model (Done)
-   Implemented NanoGPT architecture with stateful KV-cache and AdamW trainer.

### Phase 2.1: Hard Constraint Baseline (2GB RAM Reality Check) (In Progress)
-   **Goal:** Ensure the model is runnable on a low-resource machine.
-   **Status:** Initial baseline established; investigating memory-mapped token loading.

### Phase 2.2: Narrative Quality & Control (Planned)
-   **Goal:** Improve story coherence via structural control tokens and better sampling.

### Phase 2.3: Rolling Summary Memory (Planned)
-   **Goal:** Simulate long-term memory with a fixed, small context window via rolling summaries.

### Phase 2.4: Advanced Optimizations & Deployment (Planned)
-   **Goal:** Further reduce resource usage and package the model for use.
