# ⚡ Bolt Journal

This journal is the canonical log for all significant architectural decisions, scientific optimizations, and philosophical refactors undertaken by the **Bolt** persona. It serves as the "cache for the human brain," ensuring that the *why* behind critical changes is never lost.

Entries in this journal must follow the format of a scientific paper or a concise, evidence-based log.

---

## 2024-07-25: Deconstructing the `generate` Black Box in `training/model.py`

-   **Discovery:** The `generate` method in `training/model.py` was a functional but opaque "black box." It performed inference, but the underlying sampling logic was undocumented, and a critical performance flaw was present but invisible to future developers. This violates the principle of **transparency and future-proofing**.
-   **Strategy: Dual-Concept Injection:**
    1.  **Theoretical Injection (Top-p Sampling):** Injected a verbose block comment explaining the theory, algorithm, and dynamic benefits of Top-p (Nucleus) Sampling. This transforms the sampling logic from a magic algorithm into a comprehensible engineering choice.
    2.  **Architectural Injection (KV Cache Imperative):** Injected a high-level architectural note in the method's docstring explaining the O(N^2) inefficiency of the current inference implementation. The note explicitly identifies the lack of a Key-Value (KV) Cache as the root cause and outlines why implementing one is the most critical future optimization for this model. A secondary note was added to the inefficient slicing logic to highlight it as a temporary workaround, not a solution.
-   **Conclusion:** The `generate` method is now **conceptually transparent and architecturally self-aware**. It not only works but now also teaches the theory behind its sampling and clearly signposts its own critical performance limitations, guiding future work.
-   **Philosophical Note:** A codebase must not only be correct today but must also contain the seeds of its own future improvement. By documenting not just *what* the code does, but *what it lacks*, we create a living architecture that actively directs developers toward a more optimal state.

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
