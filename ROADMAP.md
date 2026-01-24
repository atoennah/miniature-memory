# Project Roadmap

This document outlines the long-term vision and development phases for the `miniature-memory` project. It serves as a guide for contributors to understand our strategic priorities.

## Phase 1: Foundational Pipeline (Complete)

-   [x] **Objective:** Establish a fully automated, end-to-end pipeline for data processing and model training.
-   [x] **Key Results:**
    -   [x] Implement `run.py` as the central orchestrator.
    -   [x] Develop scripted, reproducible data cleaning and preparation stages.
    -   [x] Integrate a configurable NanoGPT-style training module.
    -   [x] Establish a clear, non-negotiable project intent in `CONTRIBUTING.md`.

## Phase 2: Data Scaling & Quality Assurance

-   [ ] **Objective:** Systematically grow the dataset while improving its narrative quality and coherence.
-   [ ] **Key Initiatives:**
    -   [ ] **Automated Source Discovery:** Enhance the scraper to find new, high-quality sources of narrative content.
    -   [ ] **Advanced Cleaning:** Implement more sophisticated filters to detect and remove non-narrative content (e.g., ads, comments, navigation menus).
    -   [ ] **Quality Metrics:** Develop a scoring system to automatically rank the quality of raw data sources.

## Phase 3: Model Optimization & Experimentation

-   [ ] **Objective:** Improve the model's performance, efficiency, and generative capabilities under strict resource constraints.
-   [ ] **Key Initiatives:**
    -   [ ] **Hyperparameter Tuning:** Conduct systematic experiments to find the optimal training configuration.
    -   [ ] **Architectural Enhancements:** Explore minor modifications to the model architecture that could improve performance without significantly increasing its size.
    -   [ ] **Inference Speed:** Optimize the text generation process for faster output.

## Phase 4: Advanced Generation & Control

-   [ ] **Objective:** Introduce more sophisticated techniques for controlling the style and content of the generated text.
-   [ ] **Key Initiatives:**
    -   [ ] **Instruction Tuning:** Experiment with fine-tuning the model on instruction-based datasets to allow for more direct control over the output.
    -   [ ] **Style Conditioning:** Investigate methods for guiding the generation process toward specific narrative styles or themes.
