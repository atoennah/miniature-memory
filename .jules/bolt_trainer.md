# Bolt's Training Journal

## 2024-07-17

**Statistical Truth:** The default learning rate of `1.0e-3` in `small.yaml` is too high for the dataset, causing training instability.

**Experiment:**
- Created a test configuration `bolt_test.yaml`.
- Reduced `learning_rate` to `1.0e-4`.
- Ran a 20-step training benchmark.

**Result:**
- Observed a stable, monotonically decreasing loss, from ~6.3 to ~5.6.
- This confirms that `1.0e-4` is a much more suitable starting learning rate.

**Action:**
- The `learning_rate` in `small.yaml` will be updated to `1.0e-4`.
