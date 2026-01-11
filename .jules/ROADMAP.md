# Project Roadmap

This document outlines the unified development plan for the `miniature-memory` project, covering both the data pipeline and the model evolution. It serves as the single source of truth for the project's trajectory.

## Guiding Principles

-   **Constraint-First:** All development prioritizes functionality within tight resource limits (~2GB RAM).
-   **Automation:** All data processing is scripted, deterministic, and reproducible.
-   **Intent-Locked:** The project remains focused on its adult-entertainment purpose.

---

## Part 1: Data Pipeline - The Foundation

This phase focuses on building a robust, automated pipeline to continuously discover, process, and prepare high-quality training data.

### Phase 1.0: Foundations & Workflow

-   **Goal:** Establish a reproducible, collaboration-safe repository structure.
-   **Key Deliverables:**
    -   Define canonical directory structure (`dataset/`, `scraper/`, `scripts/`, etc.).
    -   Lock in ground rules: append-only raw data, scripted transformations, deterministic outputs.

### Phase 1.1: URL Discovery (Search Layer)

-   **Goal:** Maintain a growing list of candidate URLs for scraping.
-   **Process:**
    -   Use Google Search with targeted queries to discover potential story URLs.
    -   Store findings in `dataset/metadata/urls.jsonl` with a `status` field (`new`, `fetched`, `rejected`).
-   **Output:** A continuously growing manifest of potential data sources.

### Phase 1.2: Fetching & Extraction

-   **Goal:** Retrieve and extract clean, readable story text from URLs.
-   **Process:**
    -   Fetch full HTML using a headless browser to handle JavaScript-heavy sites.
    -   Use a "read mode" algorithm to extract the core narrative content, removing UI noise.
-   **Output:** Raw, extracted text stored one file per story in `dataset/raw/`.

### Phase 1.3: Deduplication & Quality Control

-   **Goal:** Prevent dataset rot and ensure high-quality inputs.
-   **Process:**
    -   Implement both exact (hash-based) and near-duplicate (similarity fingerprinting) detection.
    -   Filter out content that is too short, in the wrong language, or not narrative.
-   **Output:** Rejected content is moved to `dataset/rejected/` for auditability.

### Phase 1.4: Cleaning & Normalization

-   **Goal:** Convert raw text into a model-friendly format.
-   **Process:**
    -   Normalize whitespace, punctuation, and character encoding.
    -   Remove site-specific artifacts (e.g., watermarks, promotional text).
-   **Output:** Cleaned text files stored in `dataset/cleaned/`. Raw data is never modified.

### Phase 1.5: Preparation & Automation

-   **Goal:** Create the final training corpus and automate the entire pipeline.
-   **Process:**
    -   Concatenate cleaned files into `train.txt` and `val.txt` in a deterministic order.
    -   Develop a main `run.py` script to execute the entire pipeline (discovery through preparation).
-   **Output:** A single `train.txt` file ready for the model.

---

## Part 2: Model Evolution - From NanoGPT to Memory-Aware

This phase focuses on developing a small, efficient GPT-style model capable of generating coherent narratives under extreme memory constraints.

### Phase 2.0: Foundational Model

-   **Goal:** Implement a correct, minimal NanoGPT that trains and generates text.
-   **Key Deliverables:**
    -   Character-level tokenizer.
    -   Decoder-only Transformer architecture.
    -   Stable training loop with AdamW optimizer.
    -   Basic text generation script.

### Phase 2.1: Hard Constraint Baseline (2GB RAM Reality Check)

-   **Goal:** Ensure the model is runnable on a low-resource machine.
-   **Optimizations:**
    -   Reduce model parameters (e.g., ≤25M).
    -   Enforce a small context window (`block_size` ≤ 256).
    -   Use FP16 and `torch.no_grad()` for inference.
-   **Metric:** Peak RAM usage during inference must be well under 1GB.

### Phase 2.2: Narrative Quality & Control

-   **Goal:** Improve story coherence without increasing the context window.
-   **Techniques:**
    -   Introduce structural control tokens (e.g., `<|scene_break|>`, `<|end_of_text|>`) into the training data.
    -   Tune sampling parameters (temperature, top-p, repetition penalty) to reduce degeneration.

### Phase 2.3: Rolling Summary Memory (Core Novelty)

-   **Goal:** Simulate long-term memory with a fixed, small context window.
-   **Mechanism:**
    1.  Implement a sliding window for text generation.
    2.  As the window slides, generate a condensed summary of the text that is about to be evicted.
    3.  Prepend this summary (prefixed with a `<|summary|>` token) to the next context window.
    4.  Train the model to understand and utilize these summary tokens.
-   **Metric:** The model should maintain narrative coherence over thousands of generated tokens while RAM usage remains flat.

### Phase 2.4: Advanced Optimizations & Deployment

-   **Goal:** Further reduce resource usage and package the model for use.
-   **Techniques:**
    -   Implement a KV-cache for faster inference.
    -   Explore micro-optimizations like INT8 weight loading.
-   **Deployment:**
    -   Create a clean CLI for text generation.
    -   Provide clear documentation for deploying on low-RAM environments.
    -   Develop a robust evaluation suite to measure quality under constraint.
