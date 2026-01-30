# ⚡ Bolt Injector's Journal: The Ledger of Transferable Wisdom

This journal documents the injection of deep, conceptual knowledge into the codebase. Each entry represents a "black box" that was illuminated, turning opaque code into an educational asset.

---

### **Entry 1: Demystifying the Core Transformer Kernels**

*   **Date:** 2024-07-16
*   **Target Module:** `training/model.py`
*   **Wisdom Injected:**
    1.  **Fused Attention Kernel (`CausalSelfAttention.forward`):** Injected a detailed explanation of `F.scaled_dot_product_attention`. The previous implementation treated this as a magic function. The injection clarifies *why* it's critical for performance, covering reduced memory I/O (the core insight of FlashAttention), hardware-specific optimizations, and numerical stability. This prevents future developers from mistakenly implementing a naive, slow version.
    2.  **Nucleus Sampling (`GPT.generate`):** The top-p sampling logic was an uncommented, dense algorithm. I injected a full step-by-step "textbook" explanation of the process, from sorting probabilities to renormalizing the final distribution. This makes the generation process transparent and prevents incorrect modifications to the sampling logic.
*   **Impact:** Transformed the two most critical and algorithmically complex parts of the model from opaque kernels into well-documented, educational components. This significantly lowers the barrier for new developers to understand and contribute to the core model architecture.
