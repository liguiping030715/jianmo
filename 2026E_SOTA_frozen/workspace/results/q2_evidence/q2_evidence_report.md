# Q2 evidence completion — frozen model robustness

**Q2_EVIDENCE_COMPLETE.** The final M0 seed2718 predictor and A3 predictions remain frozen. This is an inference-only A2-valid diagnostic on six frozen M0/M1 checkpoints. No model was trained, selected or changed, and no A2 test / Attachment 3 / Attachment 4 data were accessed.

The balanced type×position×ratio grid has **36 cells × 6 checkpoints**, all on the same **715/728** valid samples. 13 rows lack three feasible contiguous native-index starts at every requested ratio; their IDs and reasons are in [factorial_summary.json](factorial_summary.json). The common cohort is a conditional subset; original 728-row validation metrics remain separate.

## Paper figures

- [Missing modality type → performance](../../figures/q2_missing_type_performance.png)
- [Missing interval position → performance](../../figures/q2_missing_position_performance.png)
- [Missing ratio/native duration → performance](../../figures/q2_missing_ratio_duration_performance.png)

Each figure has macro-F1 degradation and MAE increase panels. Positive values mean worse performance. Points are the marginal mean over the other two balanced factors, with sample SD across the three frozen seeds. The ratio denotes fraction of eligible feature positions; neither interval lengths nor positions are physical seconds. M0 and M1 remain separate in every panel.

Within this synthetic grid, text replacement gives the clearest degradation: M0 mean macro-F1 loss **0.0349** and MAE increase **0.0267**; M1 corresponding values are **0.0237** and **0.0267**. Audio and vision gap effects are close to zero on average and sometimes slightly improve a metric, so they do not support a broad claim that every missing modality harms this predictor. For M0, mean macro-F1 degradation rises from **0.0022** at 10% to **0.0129** at 30%; MAE increase rises from **0.0027** to **0.0077**. Position differences are smaller than the across-seed variation shown in the figure; no position is declared universally worst.

## Frozen M0 versus M1 (full 728-row valid, three seeds)

| Model | Clean macro-F1 mean ± SD | Clean MAE mean ± SD | S1–S6 gap macro-F1 mean ± SD | S1–S6 gap MAE mean ± SD |
|---|---:|---:|---:|---:|
| M0 | 0.5812 ± 0.0256 | 0.6060 ± 0.0250 | 0.5588 ± 0.0333 | 0.6223 ± 0.0275 |
| M1 | 0.5717 ± 0.0282 | 0.6012 ± 0.0271 | 0.5570 ± 0.0343 | 0.6187 ± 0.0306 |

The table is recomputed from the six already frozen Bridge metrics files. S1–S6 coupled modality, position and ratio and serve only as model-screening evidence. M1's earlier promotion failure is unchanged; this table does not reselect it.

## Interpretation boundary

The factorial grid varies one declared design factor while balancing the others on identical samples. It supports conditional model-response comparisons within the tested factor levels; it does not identify real-world causal effects, official Attachment 3 missingness, or an exact seconds-based duration. Text uses `[UNK]` plus frozen BERT; A/V use zeroed feature rows plus synthetic `O=0`. Native zeros and P/O remain intact outside each copied intervention.

Machine-readable cells: [factorial_cell_metrics.csv](factorial_cell_metrics.csv); marginals: [factorial_marginal_mean_sd.csv](factorial_marginal_mean_sd.csv); fixed spans: [factorial_mask_manifest.jsonl](factorial_mask_manifest.jsonl); full frozen comparison: [m0_m1_three_seed_mean_sd.csv](m0_m1_three_seed_mean_sd.csv).
