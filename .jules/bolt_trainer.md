# ⚡ Bolt's Trainer Journal

This journal logs the statistical truths, hyperparameter optimizations, and qualitative evaluations performed by Bolt (The Lead Trainer & Evaluator).

## 📊 Summary of Wins

| Date | Task | Result | Score |
|------|------|--------|-------|
| 2025-01-24 | Initial Audit & Baseline | Completed | 1.0 |

---

## 🔍 Data Audits

### [2025-01-24] Initial Baseline Audit

#### 1. Raw Validation
- **Command:** `python scripts/validate_raw.py --raw_data_dir dataset/raw`
- **Result:**
    - Passed: 67
    - Failed: 862 (Primary failure: `Failed language check`)
- **Note:** A significant portion of the raw dataset appears to be non-Indonesian content or failed the heuristic language check. This is a "Garbage In" risk that must be monitored.

#### 2. Cleaned Statistics
- **Command:** `python scripts/stats.py dataset/cleaned`
- **Result:**
    - Total Files: 929
    - Total Characters: 1,828,050
    - Total Words: 331,964
- **Language Register:**
    - Formal: 43.89%
    - Informal: 56.11%
- **Observation:** The dataset is small (~1.8M chars). The register leans informal, which is consistent with the Wattpad source.

#### 3. Training Corpus Audit
- **File:** `dataset/processed/train.txt`
- **Findings:**
    - Encoding: UTF-8 confirmed.
    - Control Tokens: `<|story_start|>` and `<|end_of_text|>` correctly placed.
    - **POLLUTION ALERT:** Detected Wattpad boilerplate ("Write stories", "Receive real-time notifications") at the start of the file.
- **Action:** Need to expand the blacklist in `scripts/clean_dataset.py` to remove this noise.

## ⚡ Config Critique

### [2025-01-24] `small.yaml` Evaluation

#### 1. Learning Rate
- **Current:** `1e-3`
- **Critique:** Potentially too aggressive for a tiny 1.8M character dataset. High risk of "Loss Spikes".
- **Proposed:** `5e-4` with cosine decay.

#### 2. Batch Size
- **Current:** `32`
- **Critique:** Safe for CPU training. No change needed for now.

#### 3. Missing Inference Config
- **Issue:** `scripts/generate.py` expects an `inference` block in the YAML, which is currently missing.
- **Proposed:** Add `inference` with `temperature: 0.8` and `top_p: 0.9`.

#### 4. Model Depth
- **Current:** 6 layers, 6 heads.
- **Critique:** Appropriate for a "Miniature" model.

#### Winning Parameters:
- `learning_rate`: `5e-4`
- `inference.temperature`: `0.8`
- `inference.top_p`: `0.9`

## 🧪 Qualitative Review

### [2025-01-24] Baseline Run (50 Steps)
- **Model:** `small.yaml` (6 layers, 384 dim)
- **Loss:** 2.75
- **Samples:**
    1. "Seorang fent moure den I povando..." (Score: 1/10)
    2. "Dia muki jata. ta mur a mere kie..." (Score: 1/10)
    3. "Dia adalah aca y a ma aa arane..." (Score: 1/10)
- **Verdict:** Repetitive mush. The model has not yet learned basic Indonesian grammar or word boundaries. 50 steps is insufficient for any coherence.
- **Next Step:** Training needs to be scaled to 5000+ steps on a GPU to see real words.
