# Roadmap: Constraint-First NanoGPT for Storytelling

This roadmap outlines the development plan for the miniature-memory project.

## Phase 0 — Foundations (Baseline, No Optimizations)
**Goal:** A correct, minimal NanoGPT that trains and generates.
**Deliverables**
- [ ] Character-level tokenizer
- [ ] Decoder-only Transformer (causal self-attention)
- [ ] Training loop (AdamW, cross-entropy)
- [ ] Basic text generation (temperature sampling)
- [ ] Single-file inference script
**Exit criteria**
- Model trains without instability
- Can generate stylistically consistent short text
- No concern yet for RAM limits

## Phase 1 — Hard Constraint Baseline (2 GB Reality Check)
**Goal:** Make inference actually runnable on a 2 GB RAM machine.
**Tasks**
- [ ] Reduce model to ≤25M parameters
- [ ] Enforce block_size ≤ 256
- [ ] FP16 inference
- [ ] `torch.no_grad()` everywhere
- [ ] Batch size hardcoded to 1
- [ ] Disable unused buffers / debug tensors
**Metrics**
- Peak RAM usage < 700 MB
- Stable generation up to 500–700 tokens (with resets)
**Exit criteria**
- No OOM on target machine
- Generation speed acceptable on CPU

## Phase 2 — Narrative Quality Under Small Context
**Goal:** Improve story coherence without increasing context length.
**Key idea:** quality through structure, not scale.
**Tasks**
- [ ] Dataset formatting (scene boundaries, consistent POV)
- [ ] Introduce control tokens (`<SCENE>`, `<POV>`, `<TONE>`)
- [ ] Train on longer contiguous samples
- [ ] Tune sampling (temperature, top-p, repetition penalty)
**Evaluation**
- Subjective: fewer incoherent jumps
- Objective: reduced repetition loops
**Exit criteria**
- Multi-paragraph output feels intentional
- Clear beginning → escalation → resolution pattern

## Phase 3 — Rolling Summary Memory (Core Novelty)
**Goal:** Fake long-term memory under fixed context.
**Tasks**
- [ ] Implement sliding window generation
- [ ] Periodically summarize old tokens into ≤40 tokens
- [ ] Prepend `<SUMMARY>` tokens before each new window
- [ ] Train model to expect summaries
**Why this matters:** This is where the project stops being “just another NanoGPT”.
**Metrics**
- Memory usage remains flat during long sessions
- Story continuity preserved across windows
**Exit criteria**
- Can generate thousands of tokens across windows
- No attention matrix blow-up

## Phase 4 — Memory & Compute Micro-Optimizations
**Goal:** Push limits without architectural changes.
**Tasks**
- [ ] KV-cache pruning (keep only recent tokens + summary)
- [ ] Reduce FFN expansion ratio where possible
- [ ] Asymmetric block design (lighter early layers)
- [ ] Optional INT8 weight loading (inference-only)
**Metrics**
- Lower RAM footprint
- Faster per-token generation
**Exit criteria**
- Sustained long generation on low-end CPU
- No degradation in narrative flow

## Phase 5 — Training Efficiency (Optional but Valuable)
**Goal:** Make training cheaper and more targeted.
**Tasks**
- [ ] Style-focused datasets (single dominant voice)
- [ ] Curriculum training (short → long scenes)
- [ ] Partial fine-tuning (freeze embeddings)
- [ ] LoRA-style low-rank adapters (optional)
**Exit criteria**
- Faster convergence
- Better stylistic consistency per epoch

## Phase 6 — Evaluation & Instrumentation
**Goal:** Measure quality under constraint, not raw power.
**Tasks**
- [ ] Track repetition rate
- [ ] Track average sentence length
- [ ] Detect degeneration loops
- [ ] Log summary drift across windows
**Outcome:** You’ll be able to say: “This model maintains narrative coherence for X tokens using Y MB of RAM.” That’s rare — and valuable.

## Phase 7 — Packaging & Deployment
**Goal:** Make it usable by other developers.
**Tasks**
- [ ] CLI interface for generation
- [ ] Config presets (tiny / small / max-safe)
- [ ] Documentation for low-RAM deployment
- [ ] Example scripts for edge / VPS / Android
**Exit criteria**
- One-command inference
- Predictable RAM usage
- Reproducible outputs

## Phase 8 — Research Extensions (Optional)
If you want to push further:
- Hierarchical summaries (summary of summaries)
- Token importance pruning
- Lightweight recurrence
- Hybrid char + micro-BPE tokenization
This is where it becomes publishable or blog-worthy.
