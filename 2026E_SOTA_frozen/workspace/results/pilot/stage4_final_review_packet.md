# Stage 4 Final Review Packet — 2026E

## A. 范围与冻结声明

本 packet 汇总 Stage 4 pilot 全部候选在 3 个 seed 下的结果与门槛判定。
- aligned 筛选决策在 RU 运行前由 reviewer interim decision 冻结：R0=reference，R0a=augmentation reference，R1=NOT_PROMOTED，R2=REJECTED，C2=C2_TEACHER_REJECTED，R1x=sensitivity diagnostic only。
- 本轮补跑 RU / unaligned_500（text 50、audio/vision 500×（74/35）、native 索引 gap mask、B1 式 modality-native 池化、无对齐网络），并仅以已冻结的 promotion rule 评估。
- 本轮未：更改晋升阈值、重调被拒候选、训练 C2 student、进入全量实验、用 A2 test/A3/A4 做选择、自动重设计模型。

## B. Clean valid 三 seed 指标

| 候选 | seed | Accuracy | macro-F1 | MAE | Pearson |
|---|---|---|---|---|---|
| R0 | 1729 | 0.6346 | 0.5990 | 0.6019 | 0.6417 |
| R0 | 2718 | 0.6195 | 0.5938 | 0.5996 | 0.6338 |
| R0 | 31415 | 0.6140 | 0.5178 | 0.6001 | 0.6332 |
| **R0 均值** | — | **0.6227** | **0.5702** | **0.6006** | **0.6363** |
| R0a | 1729 | 0.6250 | 0.5855 | 0.6128 | 0.6201 |
| R0a | 2718 | 0.6319 | 0.5777 | 0.6135 | 0.6232 |
| R0a | 31415 | 0.6223 | 0.5704 | 0.6099 | 0.6187 |
| **R0a 均值** | — | **0.6264** | **0.5779** | **0.6121** | **0.6207** |
| R1 | 1729 | 0.6360 | 0.6153 | 0.6038 | 0.6351 |
| R1 | 2718 | 0.6099 | 0.5572 | 0.6043 | 0.6331 |
| R1 | 31415 | 0.6209 | 0.5449 | 0.6134 | 0.6420 |
| **R1 均值** | — | **0.6223** | **0.5725** | **0.6072** | **0.6367** |
| R1x | 1729 | 0.6113 | 0.5604 | 0.6379 | 0.6104 |
| R1x | 2718 | 0.6044 | 0.5044 | 0.6258 | 0.6178 |
| R1x | 31415 | 0.6085 | 0.5217 | 0.6433 | 0.6127 |
| **R1x 均值** | — | **0.6081** | **0.5289** | **0.6356** | **0.6136** |
| R2 | 1729 | 0.6003 | 0.5140 | 0.6430 | 0.6378 |
| R2 | 2718 | 0.6429 | 0.6029 | 0.5924 | 0.6554 |
| R2 | 31415 | 0.6236 | 0.5708 | 0.5959 | 0.6421 |
| **R2 均值** | — | **0.6223** | **0.5626** | **0.6104** | **0.6451** |
| C2_teacher | 1729 | 0.6099 | 0.5453 | 0.5946 | 0.6391 |
| C2_teacher | 2718 | 0.6456 | 0.6158 | 0.5911 | 0.6440 |
| C2_teacher | 31415 | 0.5975 | 0.4711 | 0.5996 | 0.6298 |
| **C2_teacher 均值** | — | **0.6177** | **0.5440** | **0.5951** | **0.6377** |
| RU | 1729 | 0.6126 | 0.5690 | 0.6182 | 0.6269 |
| RU | 2718 | 0.6401 | 0.5898 | 0.5932 | 0.6429 |
| RU | 31415 | 0.6250 | 0.5770 | 0.6050 | 0.6268 |
| **RU 均值** | — | **0.6259** | **0.5786** | **0.6055** | **0.6322** |

## C. 各候选结论

| 候选 | 决策 | 依据摘要 |
|---|---|---|
| R0 | REFERENCE | 最小 B0 基线，作为对照锚点 |
| R0a | AUGMENTATION_REFERENCE | B0 增广训练参考 |
| R1 | NOT_PROMOTED | 相对参考无达标 clean/gap 增益 |
| R1x | SENSITIVITY_DIAGNOSTIC_ONLY | 未知排除后 F1 0.5289、MAE 0.6356，仅作敏感性记录 |
| R2 | REJECTED | 门控融合无达标增益；无 alpha 塌缩但不构成晋升理由 |
| C2 teacher | C2_TEACHER_REJECTED | 1/2 seed 通过，student 未训练 |
| RU | NOT_PROMOTED | 条件 A/B 均不成立（见下） |
| Q3 wrapper | DIAGNOSTIC_ONLY / PREDICTOR_NOT_PROMOTED | 局部效应弱、seed 间不稳定，未跑 A4 |

## D. RU / unaligned_500 gate 记录

| seed | gap F1 gain | gap MAE reduction | F1 退化降幅% | MAE 退化降幅% |
|---|---|---|---|---|
| 1729 | -0.0300 | -0.0108 | -110.0% | -2.6% |
| 2718 | +0.0293 | +0.0064 | -315.4% | -6.0% |
| 31415 | +0.0287 | +0.0077 | -102.9% | -22.0% |
| **三 seed 均值** | **+0.0093** | **+0.0011** | **-176.1%** | **-10.2%** |

- 条件 A（F1≥+0.02 且 MAE≥+0.03）：**False**（F1 +0.0093，MAE +0.0011）
- 条件 B（两目标退化各降≥15%）：**False**（F1 -176.1%，MAE -10.2%）
- Clean guard：**True**（F1 +0.0061，MAE -0.0017）
- Q3 附加 faithfulness gain：**不满足**（PREDICTOR_NOT_PROMOTED）
- peak RAM 约 3.1–3.3 GiB < 14 GiB；无静默截断、无秒级声明。
- **RU = NOT_PROMOTED**

## E. Q3 wrapper 诊断记录

| 模态 | dominant 计数 (seed1729/2718/31415) | mean Δ_cls | mean Δ_reg |
|---|---|---|---|
| T | 47/49/41 | +0.2761 | +0.1531 |
| A | 1/7/18 | +0.0499 | +0.2104 |
| V | 12/4/1 | -0.0107 | +0.1636 |

- interval（middle）Δ_cls +0.0191 / Δ_reg -0.0172
- matched control（early）Δ_cls -0.0016 / Δ_reg +0.0154
- 标签：DIAGNOSTIC_ONLY / PREDICTOR_NOT_PROMOTED

## F. 三 seed 与冻结决策一览

- R0：三 seed 见 B 表，作为 reference。
- R0a：三 seed 见 B 表，作为 augmentation reference。
- R1：三 seed clean F1 0.6153/0.5572/0.5449，NOT_PROMOTED。
- R1x：三 seed 敏感性诊断，不参与选择。
- R2：三 seed clean F1 0.5140/0.6029/0.5708，REJECTED。
- C2 teacher gate：1/2 passes，C2_TEACHER_REJECTED。
- RU：三 seed clean F1 0.5690/0.5898/0.5770，NOT_PROMOTED。

## G. 结论

没有任何候选通过已冻结的晋升规则；不进入全量实验，不自动重设计模型。

PILOT_NO_PROMOTED_CANDIDATE
