---
name: data-analysis
description: Audit, clean, explore, and characterize competition datasets while preserving raw data and detecting leakage or quality problems. Use when data files are part of the modeling task.
---

# Data Analysis

1. Treat `workspace/data/raw/` as immutable.
2. Inventory every file, shape, column, type, unit, and missing-value pattern.
3. Check duplicates, impossible values, outliers, encoding issues, and time ordering.
4. Identify leakage risks and target-derived features.
5. Create decision-relevant EDA only.
6. Put transformed datasets in `workspace/data/processed/`.
7. Record transformations in `workspace/analysis/data_audit.md`.

Do not silently drop rows or impute values.
