# ⚡ Bolt's Journal: Training Logs

This journal contains only statistical truths and empirical findings from my training and evaluation runs.

## Entry: Initial Learning Rate Audit

**Hypothesis:** The default learning rate of `1e-4` in `small.yaml` is too aggressive for the dataset and will result in a suboptimal loss. A more conservative `1e-5` should perform better.

**Experiment:**
1.  **Baseline Run:** Executed a 100-step training run with `training/configs/small.yaml` (`learning_rate: 1e-4`).
2.  **Experimental Run:** Executed a 100-step training run with a modified config using `learning_rate: 1e-5`.

**Results:**
*   **Baseline Loss (`lr=1e-4`):** **2.6345**
*   **Experimental Loss (`lr=1e-5`):** 3.3553

**Conclusion:** **Hypothesis REJECTED.** The more aggressive `1e-4` learning rate is demonstrably more effective for this model over a 100-step micro-train. The faster convergence leads to a significantly lower loss. The original configuration is the "Winning" configuration for short-run performance.
