# ⚡ Bolt’s Journal: The Injector’s Ledger

This ledger logs the "Transferable Wisdom" injected into the codebase. Every injection is a bridge between raw code and scientific intent.

---

### Entry 1: KV-Caching & Inference Complexity

**Injected into:** `training/model.py`

**Observation:** Pure autoregressive generation has O(N^2) complexity because the model re-processes the entire sequence for every new token.

**Conceptual Injection:**
- Implemented a stateful **KV-cache** that stores the Key (K) and Value (V) tensors for all previous tokens in the Transformer blocks.
- **Mathematical Justification:** By caching KV pairs, we reduce the multi-head attention operation to O(N) per token, as we only need to compute the Query (Q) for the *current* token and attend to the cached K/V history.
- **Stability Fix:** Implemented a **Sliding Window** mechanism for the cache. When the sequence length reaches the `block_size` limit, the cache is truncated to ensure positional embeddings remain within valid bounds while preserving recent context.
- **Impact:** Significant reduction in generation latency, especially for long sequences.

---

### Entry 2: Vectorized Data Ingestion & Memory Mapping

**Injected into:** `training/data_loader.py`

**Observation:** Data ingestion was previously limited by Python-level loops and redundant vocabulary scans.

**Conceptual Injection:**
- **Memory Mapping:** Documented the use of `np.memmap` to map the dataset directly into the process's virtual address space, enabling O(1) memory overhead for massive datasets.
- **Vectorized Ingestion:** Refactored `get_batch` to use NumPy's advanced indexing. By creating a 2D index matrix `inds = ix[:, None] + arange(T)`, we perform a single vectorized memory fetch at the C-level.
- **Persistent Metadata:** Implemented O(1) startup by caching the vocabulary and tokenizer state into a `_meta.pkl` artifact.
- **Robustness:** Maintained chunked tokenization logic to ensure the pipeline remains RAM-safe even when processing multi-gigabyte source files.

---

### Entry 3: Sampling Theory (Temperature & Nucleus)

**Injected into:** `training/model.py` and `scripts/generate.py`

**Conceptual Injection:**
- **Temperature Scaling:** Formally documented the derivation: $P_i = \exp(L_i/T) / \sum \exp(L_j/T)$. Explained the entropy-based trade-off between diverse (high T) and focused (low T) outputs.
- **Top-p (Nucleus) Sampling:** Explained how dynamic vocabulary selection based on cumulative probability $p$ discards the "unreliable tail" of the distribution, improving coherence compared to fixed Top-k.

---

**"I make the invisible visible. I turn your codebase into a masterclass."** — Bolt ⚡
