# ⚡ Bolt: Transferable Wisdom (The Injector's Ledger)

This ledger logs the technical and theoretical insights "injected" into the codebase. It serves as a master reference for future engineers to understand the deep design choices that define this project.

---

## Technical Wisdom: The Engine of Attention

### 1. Fused Attention Kernels (`training/model.py`)
- **Wisdom:** Standard PyTorch implementations of self-attention are memory-heavy ($O(T^2)$). By leveraging `F.scaled_dot_product_attention`, we tap into hardware-optimized kernels (like FlashAttention) that use tiling and recomputation to achieve $O(T)$ memory complexity.
- **Why it matters:** This allows us to scale context length without hitting VRAM/RAM limits as aggressively.

### 2. Residual Branch Initialization
- **Wisdom:** In deep Transformers, the variance of the residual stream increases at each layer. We scale the initialization of residual projection layers by $1/\sqrt{2 \times N}$ (where $N$ is the number of layers).
- **Why it matters:** This ensures numerical stability from the very first training step, preventing the model from diverging early on.

---

## Scraper Strategy: Conceptual Transparency

### 1. Heuristic Discovery (`scraper/crawler.py`)
- **Wisdom:** Web scraping in adversarial or unstructured environments requires "probabilistic rule-sets" rather than rigid parsers. We use text density (word counts) and path filtering to identify stories.
- **Why it matters:** It makes the scraper resilient to site-wide DOM changes that don't affect the core content structure.

### 2. Stealth & Dynamic Rendering (`scraper/process.py`)
- **Wisdom:** Modern Anti-Bot measures look for "static" signatures. Using Playwright for full JavaScript execution and `networkidle` state synchronization mimics human browsing behavior and ensures content visibility.
- **Why it matters:** Without this, the scraper would only capture the "shell" of modern Single-Page Applications (SPAs).
