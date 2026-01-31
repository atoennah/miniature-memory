# ⚡ Bolt's Journal (The Injector Edition)

This journal logs the "Transferable Wisdom" injected into the codebase by the Bolt persona. Every entry represents a transformation of raw code into an educational and architectural asset.

## 📓 The Master Ledger

- **Injected a deep-dive on KV-Caching in `training/model.py` because the previous developers were confusing 'Cache' with 'Buffer'.**
  - **Theory:** Explained that KV-Caching transforms O(N^2) generation into O(N) by storing history.
  - **Implementation:** Refactored `CausalSelfAttention`, `Block`, and `GPT` to return and accept `past_key_values`.

- **Added detailed mathematical explanations for the Attention scaling factor (`sqrt(d_k)`).**
  - **Why:** To clarify that scaling prevents gradient vanishing in the softmax function, stabilizing training.

- **Injected the "Philosophy of a From-Scratch GPT" header to `training/model.py`.**
  - **Goal:** To establish the model as a pedagogical resource, not just a functional tool.

- **Added a verbose explanation of the AdamW weight decay strategy in `training/trainer.py`.**
  - **Wisdom:** Differentiated between parameters that should be decayed (Linear weights) and those that shouldn't (Biases, LayerNorm, Embeddings).

- **Injected "The Geometry of Learning Rates" into the LR scheduler logic.**
  - **Theory:** Explained the roles of Linear Warmup (prevention of explosion) and Cosine Decay (exploration vs. exploitation).

- **Added "The Architecture of Stealth Crawling" to `scraper/process.py`.**
  - **Insight:** Detailed the bot-detection challenges (TLS fingerprinting, JS execution) and the use of Playwright as a counter-measure.

- **Introduced the "Content vs Boilerplate" heuristic explanation.**
  - **Insight:** Highlighted the use of `trafilatura` for robust content extraction over brittle CSS selectors.

## 🏗️ TODO: Future Architectural Injections

- [ ] **[SCALING]**: Inject notes on DistributedDataParallel (DDP) when moving to multi-GPU clusters.
- [ ] **[OPTIMIZATION]**: Add theory on FlashAttention-2 and its IO-awareness.
- [ ] **[SECURITY]**: Inject a deep-dive on sandbox isolation for scrapers to prevent RCE from malicious websites.
