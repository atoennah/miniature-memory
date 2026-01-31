# Bolt's Journal (Refactor Lessons)

## ⚡ Data Processing Modularity
- **Aha!**: The original data pipeline had cleaning logic scattered across multiple scripts (`validate_raw.py`, `clean_dataset.py`, `prepare_data.py`) with duplicated constants and non-reusable functions.
- **Refactor**: Consolidated all cleaning, normalization, and structural marking logic into a modular `processing/` package.
- **Benefit**: This makes the pipeline deterministic, testable, and allows for easy experimentation with new filters or normalization rules without touching the CLI wrappers.

## ⚡ Benchmark configuration
- **Aha!**: `benchmark.py` was failing because it didn't account for the nested structure of the YAML configuration files, which is a common source of "Silent Debt" in ML projects.
- **Refactor**: Updated the benchmark script to correctly parse nested `model` and `training` keys.
