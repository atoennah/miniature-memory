# ⚡ Bolt's Journal (Architect)

This journal documents critical, repository-specific architectural learnings.

---

### Entry 1: GELU Approximation Performance

**💡 Discovery:** The model's MLP layers use the standard `nn.GELU()` activation. My hypothesis was that replacing it with the `approximate='tanh'` version would increase throughput on a CPU-only environment.

**📊 Evidence:**
-   **Baseline (Exact GELU):** ~9351 tokens/sec
-   **Tweak (Approximate GELU):** ~6842 tokens/sec
-   **Result:** ~26.8% performance *regression*.

**🎯 Architectural Learning:**
In this specific CPU environment (PyTorch 2.3.0), the default, precise GELU implementation is significantly more optimized than the `tanh` approximation. The performance cost of the `erf` function is lower than the cost of the operations used in the approximation. This is a non-obvious finding and contradicts common performance tuning advice.

**Action:** The change was reverted. The standard `nn.GELU()` must be preserved.
