# ⚡ Bolt Injector: The Master Ledger of Logos

This ledger logs all **Conceptual Injections**, mathematical proofs, and architectural deep-dives performed by the **Bolt (The Injector)** persona.

---

## 2024-07-24: Deep Architectural Injection for `training/model.py`

-   **Conceptual Injection:** Transformed the transformer core into a masterclass of LLM theory. Added a 20-line theory header deriving the Logos of Attention.
-   **Implementation Note:** Re-implemented and documented the **Stateful KV-Cache** in `forward` and `generate`.
-   **Theory Injected:**
    -   **Attention Scaling:** Explained the mathematical necessity of `1/sqrt(d_k)` to prevent gradient death.
    -   **KV-Caching:** Detailed the spatial-temporal dynamics (O(T) vs O(T^2)).
    -   **SDPA Kernels:** Demystified the "Black Box" of `scaled_dot_product_attention` (FlashAttention-2).
    -   **Tied Weights:** Injected logic to handle shared memory in the optimizer, preventing `KeyError` during tied-weight training.
-   **Future-Proofing:** Seeded the codebase with `TODO [SCALING]` blocks for DDP, FSDP, and Activation Checkpointing.

**Transferable Wisdom:**
"In architectures with tied weights, `named_parameters()` only yields the primary name. Optimizers must filter their parameter sets against the active parameter dictionary to avoid segmentation or key errors."
