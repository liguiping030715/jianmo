# Q2 Bridge — M0 native MulT backbone

Three frozen seeds; A2 unaligned valid only. RU is the frozen native lightweight comparator. All metrics are from saved audit checkpoints; no model was rerun for this summary.

| Model | Seed | Clean Acc | Clean macro-F1 | Clean weighted-F1 | Clean MAE | Clean Pearson | Six-gap mean F1 | Six-gap mean MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RU | 1729 | 0.6126 | 0.5690 | 0.5999 | 0.6182 | 0.6269 | 0.5657 | 0.6275 |
| RU | 2718 | 0.6401 | 0.5898 | 0.6211 | 0.5932 | 0.6429 | 0.5705 | 0.6108 |
| RU | 31415 | 0.6250 | 0.5770 | 0.6084 | 0.6050 | 0.6268 | 0.5476 | 0.6180 |
| **RU mean** | — | **0.6259** | **0.5786** | **0.6098** | **0.6055** | **0.6322** | **0.5613** | **0.6188** |
| M0 | 1729 | 0.6332 | 0.6030 | 0.6284 | 0.6111 | 0.6353 | 0.5818 | 0.6168 |
| M0 | 2718 | 0.6305 | 0.5877 | 0.6154 | 0.5788 | 0.6523 | 0.5741 | 0.5980 |
| M0 | 31415 | 0.6154 | 0.5529 | 0.5879 | 0.6281 | 0.6348 | 0.5205 | 0.6521 |
| **M0 mean** | — | **0.6264** | **0.5812** | **0.6106** | **0.6060** | **0.6408** | **0.5588** | **0.6223** |

## Scenario audit

| Scenario | M0 gap F1 mean ± seed SD | M0 gap MAE mean ± seed SD | RU gap F1 mean | RU gap MAE mean |
|---|---:|---:|---:|---:|
| S1 | 0.5639 ± 0.0302 | 0.6295 ± 0.0350 | 0.5635 | 0.6187 |
| S2 | 0.5834 ± 0.0212 | 0.6027 ± 0.0225 | 0.5743 | 0.6046 |
| S3 | 0.5771 ± 0.0252 | 0.6052 ± 0.0246 | 0.5735 | 0.6070 |
| S4 | 0.5280 ± 0.0459 | 0.6423 ± 0.0319 | 0.5287 | 0.6396 |
| S5 | 0.5169 ± 0.0597 | 0.6502 ± 0.0331 | 0.5487 | 0.6378 |
| S6 | 0.5836 ± 0.0244 | 0.6041 ± 0.0241 | 0.5789 | 0.6051 |

Mean clean→gap M0 macro-F1 change: -0.0208; MAE change: 0.0155. Native-index scenarios combine modality, position and ratio; they do not identify independent factor effects.

## H11 frozen gate

M0 clean mean F1 0.5812 versus RU 0.5786; M0 MAE 0.6060 versus RU 0.6055. Six-gap means: F1 0.5588 versus 0.5613; MAE 0.6223 versus 0.6188. Joint clean bounds pass in 2/3 seeds. All resource limits pass. **H11 PASSES as an A2-valid competitiveness gate.** This does not establish superiority over RU or R1 and does not clear the A3 interface blocker.
