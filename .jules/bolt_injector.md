# ⚡ Bolt: The Injector Journal

This journal logs all conceptual injections, mathematical proofs, and architectural refactors performed by **Bolt (The Injector)**. It serves as the encyclopedia for the project's technical depth.

---

## 2024-07-24: Refactor to `transformer.py` & Logos Injection

-   **Technical Decision:** Renamed `training/model.py` to `training/transformer.py`.
-   **Justification:** The project is a pure Transformer implementation. Naming the core module `transformer.py` aligns with architectural clarity and the user's conceptual expectation.
-   **Conceptual Injection:**
    -   **Ontology of Autoregressive Modeling:** Added a file-level header explaining the decomposition of joint probability and the emergence of reasoning through next-token prediction.
    -   **Logos of the Fused Kernel:** Injected a deep-dive into `F.scaled_dot_product_attention`. Explained the mathematical derivation of the $1/\sqrt{d_k}$ scaling factor and the memory efficiency gains ($O(N)$ vs $O(N^2)$) provided by tiling/recomputation in FlashAttention.
-   **Impact:** The module is now an educational asset that explains the "Why" behind the "How," specifically addressing the "Black Box" of the fused attention kernel.

---
