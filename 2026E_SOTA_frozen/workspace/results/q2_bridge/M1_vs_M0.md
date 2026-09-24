# Q2 Bridge — M1 versus M0 observation adapter

Same seed, mask bytes, backbone width/depth/heads and optimization. Positive F1 delta or MAE reduction favors M1.

| Seed | M0 clean F1 | M1 clean F1 | Δ clean F1 | M0 clean MAE | M1 clean MAE | Δ gap F1 | Gap MAE reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 0.6030 | 0.6038 | 0.0009 | 0.6111 | 0.5906 | 0.0070 | 0.0121 |
| 2718 | 0.5877 | 0.5602 | -0.0275 | 0.5788 | 0.5810 | -0.0125 | 0.0005 |
| 31415 | 0.5529 | 0.5511 | -0.0018 | 0.6281 | 0.6321 | 0.0001 | -0.0017 |

| Scenario | M1−M0 gap macro-F1 | M0−M1 gap MAE |
|---|---:|---:|
| S1 | -0.0136 | 0.0060 |
| S2 | -0.0075 | 0.0023 |
| S3 | -0.0066 | 0.0009 |
| S4 | 0.0033 | 0.0025 |
| S5 | 0.0221 | 0.0071 |
| S6 | -0.0085 | 0.0031 |

Across three seeds and six scenarios: clean ΔF1 -0.0095, clean MAE increase -0.0048, gap F1 gain -0.0018, gap MAE reduction 0.0036. Clean guards pass. Gap F1 requires +0.010 or MAE reduction +0.015, 2/3 seed direction and 4/6 scenario direction. Observed F1 direction: 2/3 seeds and 2/6 scenarios; MAE direction: 2/3 seeds and 6/6 scenarios. **H12 FAILS**. The adapter is not promoted.
