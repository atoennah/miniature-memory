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

## 2024-05-22: KV-Cache Implementation & Ontological Refactor

-   **Hypothesis:** Implementing a Key-Value (KV) cache for the `generate` method will eliminate redundant computations of past token representations, resulting in a significant increase in inference throughput, especially for longer sequences.
-   **Methodology:**
    -   Measured baseline generation throughput for 200 tokens using a dedicated benchmark script (`benchmark_gen.py`).
    -   Refactored `GPTConfig` to a `@dataclass` for structural clarity.
    -   Modified `CausalSelfAttention`, `Block`, and `GPT` modules to support stateful KV-cache propagation.
    -   Optimized `GPT.generate` to process only the latest token per iteration after the initial prompt.
-   **Results:**
    -   **Baseline Throughput (200 tokens):** 44.42 tokens/sec
    -   **Optimized Throughput (200 tokens):** 147.30 tokens/sec
    -   **Performance Gain:** ~3.3x speedup.
-   **Conclusion:** The hypothesis was **CONFIRMED** with a substantial margin. The KV-cache is a critical optimization for autoregressive generation, transforming the computational complexity from $O(N^2)$ to $O(N)$ with respect to the generated sequence length.
-   **Philosophical Note:** The refactor to `@dataclass` aligns the code with the Principle of Ontological Clarity, making the model's configuration a first-class, type-safe entity. The implementation of KV-cache respects the Teleological Alignment of the `forward` pass, ensuring it only performs the necessary work for the given input.
