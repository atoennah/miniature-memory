# ⚡ Bolt's Journal: The Injector's Ledger

This ledger records the technical depth and conceptual clarity "injected" into the codebase.
Each entry represents a transformation of a "Black Box" module into an educational asset.

---

## 🏗️ Technical Injections

### [2024-10-27] - Scraper & Data Pipeline Deep-Dive

**Module:** `scraper/crawler.py`
- **Injection:** The Philosophy of Heuristic Discovery.
- **Wisdom:** Explained why probabilistic discovery is superior to rigid CSS selectors in the "wild web." Documented the word-count signature (3-25 words) used to promote story titles over navigation links.

**Module:** `scraper/process.py`
- **Injection:** Stealth Strategy & Content Distillation.
- **Wisdom:** Formalized the dual-pronged approach of Playwright (Stealth Emulation) and Trafilatura (Distillation). Documented the importance of the `networkidle` state for rendering dynamic content.

**Module:** `scraper/storage.py`
- **Injection:** The Logos of Canonical Data Storage.
- **Wisdom:** Defined the `{TIMESTAMP}__{SRC}__{CID}__{HASH}.txt` naming convention. Explained how SHA-256 content hashes act as the ultimate guard against dataset duplication and overfitting.

**Module:** `training/data_loader.py`
- **Injection:** The Logos of Memory-Mapped Datasets.
- **Wisdom:** Documented the O(1) random access and O(N) memory efficiency provided by `np.memmap`. Explained the necessity of the two-pass processing architecture (Vocab building vs. Tokenization) for handling datasets larger than system RAM.

**Module:** `scraper/search/discover.py`
- **Injection:** The Discovery Protocol.
- **Wisdom:** Documented the rationale for using DuckDuckGo Search (DDGS) over traditional APIs. Highlighted the "Scout" role of this module in seeding the manifest.

---

## 🔮 Future Injections (TODO)

- [ ] **Distributed Training:** Inject theory on `DistributedDataParallel` and gradient synchronization when scaling to multi-GPU clusters.
- [ ] **Transformer Kernels:** Document the fused kernels in `F.scaled_dot_product_attention` and their memory complexity. (Partially done in `model.py`).
- [ ] **Language Guards:** Inject mathematical theory on language detection (e.g., FastText or 3-gram overlapping) for filtering non-target language content.

---
*"I make the invisible visible. I turn your codebase into a masterclass."* — **Bolt (The Injector)** ⚡
