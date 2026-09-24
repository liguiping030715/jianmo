# Decision-relevant EDA

Machine-readable outputs: `workspace/results/data_audit/eda_summary.csv`, `deep_checks.json`, `label_statistics.json`, `zero_statistics.json`, and the per-file inventory. The frame contact sheet is an audit artifact, not a predictive result. No model was trained.

- **Length:** text attention-mask means 24.65 train and 25.59 valid of 50 tokens. Unaligned audio stated means 147.29 train /154.11 valid of 500; unaligned vision stated means 93.25/98.69, but these understate observed support for 618/141 samples respectively. Aligned audio last-nonzero means 23.65/24.59 of 50.
- **Padding/zero:** aligned audio all-zero-vector fraction is 54.75% train /52.95% valid; aligned vision 57.21%/55.52%. Unaligned audio 70.54%/69.18%, largely right padding; unaligned vision 79.18%/77.94%, but the length-field mismatch prevents treating this as a pure padding rate. Text continuous features contain no all-zero rows even outside attention validity.
- **Labels:** train classes 967 Negative, 758 Neutral, 1,670 Positive; valid 206/184/338. Intensities span [-3,3], with 23/22 unique stored values. Valid has a slightly larger Neutral share; no threshold was tuned.
- **Feature scale:** observed nonzero audio entries have mean 2.33 train and 2.30 valid, SD 22.15/21.71 in aligned; vision means -0.897/-0.856, SD 2.59/2.55. These summaries mix the 74 or 35 channels and do not imply unit-level physical meanings. Standardization must fit train only and be channel-aware.
- **Train/valid comparison:** analogous length, class and feature-scale statistics are close enough to support a normal validation audit, but no formal shift test was run because the critical mask/interface problems dominate the next decision. Official splits are source-video disjoint.

No decorative plot was produced. The numeric tables directly inform mask, version and split choices.
