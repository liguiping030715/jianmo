# E1 prediction comparison — Stage 3D

All values use A2 aligned_50 valid with frozen seed-1729 audit masks. R1, R0a and E0 were read only; no A2 test/A3/A4 access.

| Model | Clean Accuracy | Clean macro-F1 | Clean MAE | Clean Pearson |
|---|---:|---:|---:|---:|
| R0a (frozen) | 0.6250 | 0.5855 | 0.6128 | 0.6201 |
| R1 (frozen) | 0.6360 | 0.6153 | 0.6038 | 0.6351 |
| E0 (frozen, rejected) | 0.6058 | 0.4963 | 0.6106 | 0.6719 |
| E1-full | 0.6030 | 0.4908 | 0.6225 | 0.6270 |
| E1-local-only | 0.6319 | 0.5678 | 0.6170 | 0.6377 |
| E1-consistency-only | 0.6058 | 0.4705 | 0.6174 | 0.6236 |

E1-full versus frozen R1: clean ΔF1 -0.1246, clean ΔMAE +0.0187. The first-seed non-catastrophic F1 boundary is −0.0500; the final clean guard is −0.0150.

## Paired gap comparison: E1-full minus R1

| 情景 | 可注入样本 | gap F1 gain | gap MAE reduction | F1 退化改善 | MAE 退化改善 |
|---|---:|---:|---:|---:|---:|
| S1 | 728 | -0.1364 | -0.0254 | -0.0118 | -0.0067 |
| S2 | 724 | -0.1271 | -0.0190 | -0.0026 | -0.0009 |
| S3 | 700 | -0.1283 | -0.0176 | +0.0083 | +0.0006 |
| S4 | 722 | -0.0832 | -0.0463 | +0.0442 | -0.0288 |
| S5 | 703 | -0.1355 | -0.0478 | -0.0012 | -0.0297 |
| S6 | 713 | -0.1239 | -0.0188 | +0.0084 | -0.0001 |
| **六情景均值** | — | **-0.1224** | **-0.0291** | **+0.0076** | **-0.0109** |

Positive gap F1 gain and MAE reduction favor E1. Both six-scenario means are negative, so no predictive robustness effect reaches the frozen thresholds (+0.010 F1 or +0.015 MAE).
The 2/3-seed criterion cannot be assessed because the frozen first-seed stop prevents later runs.

Run records: `workspace/experiments/stage3d_runs/E1_*_seed1729/`. All three have config, mask copy, epoch log, valid predictions, failure log, environment hashes, checkpoint and metrics.

Q2_E1_MECHANISM_FAILED
