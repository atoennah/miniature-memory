# ⚡ Bolt (The Injector)'s Journal: Transferable Wisdom

This document is the master ledger of conceptual injections and deep architectural insights. Its purpose is to record high-value technical context that is transferable across modules and future projects. Each entry represents a piece of "Aha!" knowledge that makes the invisible visible.

---

### Entry 1: The Ubiquity of Fused Kernels
**Insight:** The use of `F.scaled_dot_product_attention` in `model.py` is more than a convenience. It represents a fundamental shift in high-performance computing for deep learning.
**Transferable Wisdom:** Always prefer fused, high-level library functions (like PyTorch's fused attention) over manual, sequential implementations. They are not just faster; they are fundamentally more memory-efficient because they prevent the materialization of large intermediate data structures (e.g., the (T, T) attention matrix). This principle holds true for CPUs and is paramount on GPUs.

### Entry 2: The Adaptive Nature of Top-p (Nucleus) Sampling
**Insight:** The Top-p sampling algorithm in `model.py` provides a robust and adaptive method for text generation that balances creativity and coherence.
**Transferable Wisdom:** When designing generative systems, avoid fixed or overly simple decoding strategies. Top-p sampling is a powerful technique because its "nucleus" of candidate tokens adapts to the model's confidence at each timestep. In high-certainty scenarios, it behaves like greedy decoding; in low-certainty scenarios, it allows for creative exploration within a bounded, high-probability space. This adaptability makes it superior to pure temperature sampling for most creative text generation tasks.
