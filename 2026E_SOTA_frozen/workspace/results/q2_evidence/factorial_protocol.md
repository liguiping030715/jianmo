# Q2 frozen-checkpoint factorial robustness protocol

Specified before inference-only factor scoring. Use **A2 unaligned valid only** and all six already frozen M0/M1 checkpoints (seeds 1729, 2718, 31415). No fitting, model selection, threshold adjustment, A2 test, Attachment 3 or Attachment 4 access. The final deployed predictor remains M0 seed2718; these are supplementary valid-set diagnostics.

## Balanced design

- **Synthetic missing-modality type:** text, audio, vision, and simultaneous audio+vision.
- **Native-index position:** earliest, median and latest feasible contiguous interval within the modality's eligible `P=1,O=1` support. For audio+vision, apply each anchor independently in the two native streams; this is not a cross-modal timestamp match.
- **Requested duration ratio:** 10%, 20% or 30% of each modality's eligible support count. Requested length is `max(1, round(ratio × support count))`, with Python's fixed rounding rule. Record realized support fraction and native index length per sample. This is a relative feature-index duration, not seconds or an official missing rate.

For every sample/modality/ratio, enumerate starts of contiguous intervals consisting entirely of eligible native positions. Require at least three feasible starts so early/middle/late are distinguishable. Fix the **same complete-case valid cohort** across all 36 type×position×ratio cells and both model families/seeds. Report excluded IDs and reason; this is a descriptive conditional cohort and does not replace the full 728-row valid metrics. No cell-specific sample selection.

Synthetic operator matches the frozen Bridge gap policy: replace affected text input IDs by `[UNK]` and rerun the pinned frozen BERT; set affected A/V feature rows to zero; set only affected synthetic `O=0`; leave frozen `P`, raw data and native unknown states unchanged. M0 consumes zeroed numeric content but ignores O; M1 consumes O and excludes injected unavailable keys. This is a declared model stress test, **not** recovery of the official hidden missing mask.

Score Accuracy, macro-F1, weighted-F1, MAE and Pearson for clean and every cell. Plot `clean macro-F1 − gap macro-F1` (larger means more degradation) and `gap MAE − clean MAE` (larger means more degradation). Marginalize each factor over the other two on the identical cohort. Display mean ± **sample SD across the three frozen seeds**, not uncertainty about the population. Keep M0 and M1 separate. Save all per-cell seed metrics and the mask manifest.

Also recompute a separate M0-versus-M1 three-seed mean±sample-SD table from **existing frozen 728-row clean and S1–S6 Bridge audit metrics**. The S1–S6 scenarios remain coupled and are not used to infer factor effects. No model promotion or reselection follows these diagnostics.

`Q2_EVIDENCE_COMPLETE` requires complete finite 36×6 cell metrics on one common cohort, checksum-verified checkpoints and preprocessing, three reproducible paper figures, and the frozen three-seed comparison. Any missing cell or failed provenance check blocks completion.
