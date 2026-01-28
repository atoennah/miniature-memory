#  Bolt Injector's Master Ledger of Transferable Wisdom

This document records high-value conceptual injections and clarifications made to the codebase. Its purpose is to create an "institutional memory" of *why* certain complexities exist and how they were demystified.

---

### Session: Deconstructing the Core GPT Model (`training/model.py`)

1.  **Wisdom Injected:** Clarified the architectural blueprint of the Transformer in `GPTConfig`.
    *   **Reasoning:** The hyperparameters (`n_layer`, `n_head`, `n_embd`, etc.) were being used as "magic numbers." I injected a detailed explanation of how these parameters collectively define the model's capacity and computational graph. This prevents future cargo-culting of configurations and encourages first-principles thinking when scaling the model.
    *   **Impact:** New developers can now understand the direct consequences of modifying the model's shape.

2.  **Wisdom Injected:** Deconstructed the Nucleus (Top-P) Sampling algorithm in `GPT.generate`.
    *   **Reasoning:** The text generation logic was a black box. The sequence of sorting, cumulative summing, and masking is non-trivial and was completely undocumented. I added a step-by-step narrative explaining the theory and implementation, turning the code into a textbook example of modern sampling strategies.
    *   **Impact:** Prevents incorrect modifications to the sampling logic and provides a clear, educational resource for anyone working on text generation.
