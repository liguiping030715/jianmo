# Stage 4 Pilot 晋升决策

冻结状态：aligned 候选决策（R0/R0a/R1/R2/C2/R1x）沿用 reviewer interim decision，未在本轮改动。
本轮新增 RU / unaligned_500 与 Q3 wrapper 小诊断的判定。

## 1. 决策总表

| 候选 | 说明 | 决策 |
|---|---|---|
| R0 | R0 / B0 最小基线（位置编码 + P 感知均值池化） | **REFERENCE** |
| R0a | R0a / B0 增广训练参考 | **AUGMENTATION_REFERENCE** |
| R1 | R1 / B1（P/O 状态嵌入 + 内容池化 + 状态摘要） | **NOT_PROMOTED** |
| R1x | R1x / B1 未知排除敏感性诊断 | **SENSITIVITY_DIAGNOSTIC_ONLY** |
| R2 | R2 / C1（门控融合） | **REJECTED** |
| C2_teacher | C2 教师（B1 级，用于蒸馏门槛） | **C2_TEACHER_REJECTED** |
| RU | RU / unaligned_500（text50, audio/vision500, native 索引, 无对齐） | **NOT_PROMOTED** |
| Q3 wrapper | aligned B1 解释包装小诊断 | **DIAGNOSTIC_ONLY / PREDICTOR_NOT_PROMOTED** |

## 2. RU gate 机械判定（pilot_plan.md RU promotion 规则）

**条件 A — paired 两目标提升（需同时满足）**
- gap macro-F1 提升：+0.0093，门槛 ≥ +0.02 → **不满足**
- gap MAE 降低：+0.0011，门槛 ≥ +0.03 → **不满足**
- 条件 A 结果：**False**

**条件 B — gap 退化幅度各降 ≥15%（需同时满足）**
- F1 退化降幅：-176.1%，门槛 ≥ +15% → **不满足**
- MAE 退化降幅：-10.2%，门槛 ≥ +15% → **不满足**
- 条件 B 结果：**False**

**Clean 守护（相对 B1）**
- clean F1 变化：+0.0061（门槛 ≥ −0.02）→ **满足**
- clean MAE 变化：-0.0017（门槛 ≤ +0.05）→ **满足**
- clean guard：**True**

**附加条件 — Q3 wrapper 上相对 aligned 的稳定 faithfulness gain**
- Q3 小诊断结论为 PREDICTOR_NOT_PROMOTED：局部效应弱且 seed 间不稳定，未显示稳定增益 → **不满足**。
- 按 reviewer 指令，RU 若需施加 wrapper，须先完成经评审的 unaligned_500 explanation-interface amendment；本轮 RU 数值条件已不通过，不进入该步骤。

**资源/语义约束**
- peak RAM 约 3.1–3.3 GiB < 14 GiB：满足；未做静默截断、未做秒级声明。

**RU 判定：条件 A、B 均不成立 → RU = NOT_PROMOTED。**

## 3. C2 teacher gate

- 状态：**C2_TEACHER_REJECTED**；要求 2 个 seed 通过，实际 1 个。
- 仅 seed 2718 通过；seed 1729、31415 教师 F1 低于 B0。
- student 未训练（student_trained=False）。

## 4. R2 gate 干预

- R2 各 seed alpha_diagnostics 的 collapse_fraction 均为 0.0（无 >0.95 的模态塌缩），门槛干预未触发；
- 但 R2 clean/gap 相对参考无达标增益，决策维持 **REJECTED**，不重调、不重训。

## 5. 最终结论

R1、R2、RU 均未通过晋升，C2 教师门槛被拒，Q3 包装不晋升。R0 为参考、R0a 为增广参考、R1x 仅为敏感性诊断。

**PILOT_NO_PROMOTED_CANDIDATE**
