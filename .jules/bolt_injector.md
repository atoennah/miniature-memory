# 📓 Bolt's Journal: The Injector

This ledger contains Transferable Wisdom—conceptual clarifications and deep architectural notes injected into the codebase. Each entry serves as a permanent record of *why* a complex piece of logic exists, ensuring that the invisible intent behind the code is made visible and permanent.

---

### Entry 1: Demystifying the Core Architecture and Generation Logic

**Location:** `training/model.py`

**Injection Summary:**
1.  **`GPTConfig` Blueprint:** Injected a comprehensive docstring into the `GPTConfig` class. This transforms the architectural hyperparameters from a simple list of variables into a detailed blueprint, explaining the role and trade-offs of `block_size`, `n_layer`, `n_head`, and `n_embd`. This is foundational knowledge for any developer tuning or scaling the model.

2.  **The Art of Nucleus Sampling:** Injected a deep-dive explanation into the `generate` method. The comment block clarifies the hierarchy of text generation strategies, from the naive Greedy Search to the more nuanced Temperature Sampling, and finally to the adaptive Nucleus (Top-p) Sampling. This provides the crucial theoretical context for why nucleus sampling is the preferred method for balancing creativity and coherence.

**Transferable Wisdom:**
- A model's configuration class is the first and most important piece of documentation. If a developer doesn't understand the levers they can pull, they cannot effectively pilot the machine.
- The most complex algorithm in a system (in this case, nucleus sampling) must never be a "black box." Leaving it undocumented creates a knowledge silo and a future point of failure. By illuminating the logic, we turn a potential liability into an educational asset.
