# Stage 3C Review Packet — 2026E（独立 H9 测试）

## A. 范围与冻结
- 训练前冻结 `stage3c_pilot_plan.md`：clean guards（ΔF1≥−0.015、ΔMAE≤+0.025）、
  有用效应（gap F1 gain≥+0.010 或 MAE red≥+0.015）、2/3 seed、4/6 scenario、
  local 响应/非模板/unknown 保留。
- E0 与 frozen R1 **唯一**差异为 temporal pooling；未重清洗、未改预处理、未救 D0/C1/C2/RU、
  未加 Transformer/MoE/distillation/reconstruction、未加全局 gate、A2/A3/A4 不参与。
- 数据/BERT/mask/scalers/S1–S6/seeds/λ/optimizer/LR/epoch/patience 全部复用 pilot_stage4。

## B. 结果（三 seed）
| seed | Δ macro-F1 | Δ MAE | 分类 |
|---|---|---|---|
| 1729 | -0.1191 | +0.0068 | D |
| 2718 | -0.0015 | -0.0202 | B |
| 31415 | +0.0118 | -0.0261 | A |
| **均值** | **-0.0363** | **-0.0132** | — |

- Clean guard F1：**False**（-0.0363）；
  Clean guard MAE：**True**（-0.0132）。
- Gap：F1 gain -0.0331、MAE red +0.0098；
  scenario 正向 F1 0/6、MAE 5/6；有用效应 **False**。
- Local 机制：w_gap=0、边界过渡、unknown 保留小权重、非静态位置模板（符合机制要求）。

## C. A/B/C/D 与任务依赖判定
- per seed：1729=D、2718=B、31415=A。
- B 稳定（≥2 seed）：**False**；
  C 稳定：**False**。
- 不满足稳定 B/C，不输出 TASK_DEPENDENT_TEMPORAL_AGGREGATION_HYPOTHESIS。

## D. 结论
local temporal pooling 的机制代码正确并对缺失响应，但训练高方差（seed1729 F1 崩溃），
clean F1 guard 失败、gap 无达标有用效应、跨 seed 方向不一致。不晋级，不进入全量实验，
不自动重设计模型。回归方向的微弱、不稳定信号仅作记录，不构成 E1 的充分依据。

E0_REJECTED
