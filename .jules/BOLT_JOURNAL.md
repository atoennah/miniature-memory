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

## 2025-01-29: Persistent Metadata and Vectorized Data Ingestion

-   **Hypothesis:** The redundant scanning of training data for vocabulary construction and the loop-based batch extraction in `DataManager` represent significant "Empirical Friction." Persistent metadata and vectorized indexing will drastically improve efficiency.
-   **Methodology:**
    1.  Introduced `{data_path}_meta.pkl` to store vocabulary and dataset statistics.
    2.  Modified `DataManager` and `scripts/generate.py` to bypass full data scans if artifacts exist.
    3.  Refactored `get_batch` using NumPy advanced indexing to eliminate Python-level sequence slicing loops.
-   **Results (Empirical Verification):**
    -   **DataManager Initialization:** 0.5433s → 0.0006s (~900x speedup).
    -   **Batch Extraction (get_batch):** 1.675ms → 0.305ms (~5.5x speedup).
-   **Conclusion:** The hypothesis is confirmed. The elimination of redundant computation and the shift to parallelized data extraction provide a massive boost to developer productivity and training throughput.
-   **Philosophical Note:** This refactor honors the **Principle of Least Action** and ensures **Ontological Clarity** by separating one-time preprocessing artifacts from runtime logic.

---

## 2025-01-29: Robust Optimizer Construction for Tied Weights

-   **Discovery:** Identified a "Conceptual Rot" in `Trainer._build_optimizer` where weight-tying (e.g., `lm_head` and `wte`) caused a `KeyError: 'lm_head.weight'` because `named_parameters()` only returns one name for shared tensors.
-   **Action:** Refactored the parameter filtering logic to dynamically align the `decay` and `no_decay` sets with the actual keys present in the model's `named_parameters()`.
-   **Conclusion:** The trainer is now resilient to weight-tied architectures, satisfying the **Principle of Robustness** and ensuring seamless compatibility with standard GPT-style implementations.
