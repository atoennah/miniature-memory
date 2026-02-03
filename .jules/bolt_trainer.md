# ⚡ Bolt’s Trainer Journal: Data Audit & Config Score

This journal logs the results of data audits, hyperparameter trials, and qualitative scores.

---

### Entry 1: Data Audit - Wattpad Noise Removal
**Date:** 2024-07-24
**Status:** SUCCESS

**Observation:** Found 81 occurrences of a 4-line Wattpad promotional block in the concatenated `train.txt`. This boilerplate ("Write stories...", "Whatever story you want to tell...") is non-narrative noise that dilutes the training signal.

**Action:** Updated `scripts/clean_dataset.py` with a `NOISE_KEYWORDS` filter. Also improved the script to delete existing files in `dataset/cleaned/` if the new cleaning results in an empty string (preventing stale data).

**Result:** Verified 0 occurrences of the targeted noise in the final corpus. Total concatenated files reduced from 929 to 848.

---

### Entry 2: Micro-Train Trial [Loss: 2.66]
**Date:** 2024-07-24
**Config:** `small_optimized.yaml`
**Steps:** 100
**Final Loss:** 2.6627

**Critique:**
- **Learning Rate:** 5e-4 with 10-step warmup. The model was stable but 100 steps is insufficient for coherence.
- **Batch Size:** 64. Efficient for CPU/GPU.
- **Qualitative Score:** 2.5/10. "Repetitive Mush" detected. The model has learned letter distributions but no grammar or vocabulary.

**Statistical Truth:** 1 epoch (approx. 100 steps at batch 64, block 256) on this 1.8MB dataset is enough to reach a loss of ~2.7, but not enough for readability.

**Recommendation:** Increase `max_steps` to 1000 and `lr_decay_iters` to 1000.
