# ⚡ Bolt Journal

This journal is the canonical log for all significant architectural decisions, scientific optimizations, and philosophical refactors undertaken by the **Bolt** persona. It serves as the "cache for the human brain," ensuring that the *why* behind critical changes is never lost.

Entries in this journal must follow the format of a scientific paper or a concise, evidence-based log.

---

## 2024-07-24: Conceptual Injection for `training/model.py`

-   **Discovery:** The core transformer logic in `training/model.py` was a functional "black box." It worked, but it did not teach. The mathematical and architectural principles were implicit, violating the core philosophy that code should be a living, educational document.
-   **Strategy:** Performed a deep, multi-level conceptual injection to transform `training/model.py` into a pedagogical asset. This included adding a file-level header on its philosophical purpose, a verbose block comment in `CausalSelfAttention` deriving the attention formula, narrated tensor transformation comments, and architectural docstrings for all major classes.
-   **Conclusion:** The module is no longer merely functional; it is **conceptually transparent**.
-   **Philosophical Note:** By embedding the "why" (the theory) directly alongside the "how" (the code), we reduce cognitive overhead, accelerate onboarding, and create a more resilient and maintainable system.

---

## 2024-05-21: Deferred Imports in `run.py`

-   **Hypothesis:** Deferring the import of pipeline modules in `run.py` until they are explicitly needed will significantly reduce the script's startup time and memory footprint.
-   **Methodology:** Measured the execution time of `run.py` with all stages skipped, both before and after moving `import` statements into their corresponding conditional blocks.
-   **Results:**
    -   **Before:** 4.819s
    -   **After:** 0.409s
    -   **Change:** 91.5% reduction in startup overhead.
-   **Conclusion:** The hypothesis was confirmed. The deferred-import pattern is a foundational optimization that aligns the codebase with its goals of minimalism and efficiency.
-   **Philosophical Note:** The code now more closely follows the Principle of Least Action, loading only the logic necessary for the task at hand.

---

## Entry: Initial Learning Rate Audit

-   **Hypothesis:** The default learning rate of `1e-4` in `small.yaml` is too aggressive for the dataset and will result in a suboptimal loss. A more conservative `1e-5` should perform better.
-   **Methodology:** Executed two 100-step training runs, one with the baseline `1e-4` learning rate and one with `1e-5`.
-   **Results:**
    -   **Baseline Loss (`lr=1e-4`):** 2.6345
    -   **Experimental Loss (`lr=1e-5`):** 3.3553
-   **Conclusion:** Hypothesis **REJECTED**. The more aggressive `1e-4` learning rate is demonstrably more effective for this model over a 100-step micro-train. The faster convergence leads to a significantly lower loss.

---

## 2024-08-15: Modular Refactor of Data Processing Pipeline

-   **Discovery:** The data processing scripts were monolithic and difficult to maintain. Logic for text normalization, quality filtering, and segmentation was scattered across multiple scripts.
-   **Strategy:** Refactored the monolithic scripts into a modular `processing/` package. Introduced `TextNormalizer`, `QualityFilter`, `Segmenter`, and a `CleaningPipeline` orchestrator.
-   **Conclusion:** The pipeline is now highly modular, testable, and maintainable. Scripts in `scripts/` are now thin wrappers around these core modules.
-   **Philosophical Note:** Separation of concerns is not just for software architecture; it's a requirement for scalable data science.

---

## 2024-08-20: KV-Cache Implementation & Performance Proof

-   **Hypothesis:** Implementing a Key-Value (KV) cache for transformer inference will significantly increase generation throughput by avoiding redundant calculations of past token representations.
-   **Methodology:** Implemented a stateful KV-cache in `training/model.py` and updated `scripts/generate.py`. Benchmarked tokens/sec before and after implementation on a CPU-only environment.
-   **Results:**
    -   **Baseline (No Cache):** ~44 tokens/sec
    -   **Experimental (With KV-Cache):** ~197 tokens/sec
    -   **Improvement:** ~4.4x speedup.
-   **Conclusion:** Hypothesis **CONFIRMED**. The KV-cache is a transformative optimization for real-time generation.
-   **Philosophical Note:** We value empirical performance. A 4x speedup is not just a number; it's the difference between a tool and a toy.

---

## 2024-08-22: Tied Weight Optimization & Bug Fix

-   **Discovery:** The training pipeline crashed with `KeyError: 'lm_head.weight'` when building the optimizer. This was caused by the model using tied weights between the token embeddings and the output head.
-   **Strategy:** Updated the `Trainer` to filter the parameter sets (decay vs. no_decay) against the actual `named_parameters()` of the model. This ensures that even when weights are tied (and thus only appear once in `named_parameters()`), the optimizer configuration remains valid.
-   **Conclusion:** The training pipeline now robustly handles models with tied weights, a common practice for reducing parameter count in constrained environments.

---

## 2024-08-25: Indonesian Erotica Landscape & Source Discovery

-   **Discovery:** High-quality Indonesian narrative data is difficult to find through standard datasets. Many sources are either blocked or highly polluted with gambling ads.
-   **Strategy:** Conducted a forensic analysis of the Indonesian internet landscape. Identified Wattpad Indonesia as the primary "Big Fish" source and established a "Golden Tags" list for targeted discovery. Created a "Pollution Blacklist" to protect the dataset from low-quality/educational noise.
-   **Conclusion:** We now have a clear, prioritized data acquisition strategy for the Indonesian language register.
