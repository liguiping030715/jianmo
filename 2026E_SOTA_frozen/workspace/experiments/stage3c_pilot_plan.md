# Stage 3C Pilot Plan — Independent H9 Test（训练前冻结）

> 本文件在看到任何 E0 结果**之前**写定并冻结。E0 训练/评估期间不得修改门槛、模型或阈值。

## 1. 科学问题与唯一架构差异

- H9：局部连续缺失是否应在时间池化**之前**以 position-level 状态处理，而不是池化后给整个模态一个全局权重？
- E0 = **frozen R1/B1 架构 + local state-aware temporal pooling**。
- 相对 frozen R1 的**唯一**差异：content temporal pooling 由 masked-mean 换为 local-scorer + masked-softmax 加权；state summary 保留不变。
- 明确保留 R1：modality encoders（proj）、位置参数、P/O tri-state 嵌入、**shared multimodal fusion**、classification head、regression head、losses（CE + λ·Huber）、optimizer、λ、train/valid 协议。
- **不使用** D0 task-specific fusion；不重清洗；不改预处理；不救 D0/C1/C2/RU；不加 Transformer/MoE/distillation/reconstruction；不加全局模态 gate；A2 test/A3/A4 不参与选择。

## 2. E0 pooling 规格

对每个模态 m、位置 t：

```
z_mt  = gelu(proj_m(x_mt) + p_embed(P_mt) + o_embed(O_mt) + pos_t)
score_mt = local_scorer_m(z_mt)              # Conv1d kernel=3，含邻近上下文，非 Transformer
w_mt  = masked_softmax(score_mt, content_mask)
content_m = sum_t w_mt z_mt
state_summary_m = masked_mean(gelu(state_mt), P_mt != 0)   # 与 R1 完全相同
h_m   = content_m + state_summary_m
# 之后 shared fusion / heads 与 R1 完全相同
```

状态/资格规则：
- `P=0`：已核验 padding，硬排除；
- `O=0`：已知 synthetic unavailable，排除 content；
- `P/O=2`：以显式 unknown state 保留（进入 state summary，不进 content 加权）；
- 零幅值**绝不**定义缺失。

## 3. 复用清单（R1 vs E0 仅 pooling 不同）

cleaning_2026e_v2、aligned_50、train/valid IDs（3395/728）、BERT cache 语义、scalers、
gap masks（S1–S6，同一 manifest）、seeds（1729/2718/31415）、λ=2.0526、AdamW lr1e-3 wd1e-4、
LR budget、epoch cap 12、patience 3、batch 策略。

## 4. 冻结筛选门槛

### 4.1 Clean guards（三 seed 均值，E0 − R1）
- Δ macro-F1 ≥ **−0.015**；
- Δ MAE ≤ **+0.025**。

### 4.2 有用效应（三 seed 均值，gap，paired 共同 injectable 口径）
- gap macro-F1 gain ≥ **+0.010**，**或** gap MAE reduction ≥ **+0.015**；
- 且对应 clean→gap **退化幅度改善**（不能只靠 clean 平移）。

### 4.3 一致性
- 改善方向至少 **2/3 seeds**；
- 在适用的 gap scenario 中至少 **4/6** 出现同方向；
- local weights 必须对 injected interval 有响应（gap 内权重下降）；
- weights **不得**塌缩成静态 position template（position-mean 不能与 gap 位置/场景无关）；
- unknown-retained 仍是主接口（不得靠硬删 unknown=2 获得增益）。

不得仅凭好看的 local-weight 图晋级。

### 4.4 晋级判定
- E0 同时满足 4.1、4.2、4.3 → **E0_PROMOTED**；
- clean guard 失败或无有用效应 → **E0_REJECTED**；
- 若出现“一任务改善、另一任务退化”且 ≥2 seeds 稳定 → 不直接拒绝，转 **TASK_DEPENDENT_TEMPORAL_AGGREGATION_HYPOTHESIS**（留待 E1）；
- 若接口/预算/数据受阻无法完成 → **STAGE3C_EXECUTION_BLOCKED**。

## 5. 运行顺序

1. 冻结本 plan；
2. E0 seed1729，直接对照 frozen R1 seed1729；
3. 接口/预算有效且无灾难失败后，跑 seed2718、seed31415；**seed1729 后不得改模型**；
4. 三 seed 后做 A/B/C/D 分类与 gate 判定。

## 6. 必备诊断（每 seed/scenario，机器可读）

- prediction：Accuracy、macro-F1、MAE、Pearson；
- robustness：clean→gap ΔF1、ΔMAE；
- local mechanism：gap 内平均权重、两边界附近平均权重、unknown 位平均权重、position-wise 平均权重、权重熵、seed/scenario 稳定性。

Local weights **不是** explanation。

## 7. 最终 A/B/C/D 分类

- A：分类与回归都改善；
- B：回归改善、分类退化；
- C：分类改善、回归退化；
- D：都不改善。

B 或 C 在 ≥2 seeds 稳定 → 输出 TASK_DEPENDENT_TEMPORAL_AGGREGATION_HYPOTHESIS。

## 8. 结果文件

- `workspace/models/model_cards/E0_local_temporal_pooling.md`
- `workspace/experiments/stage3c_pilot_plan.md`（本文件）
- `workspace/results/stage3c/E0_comparison.md`
- `workspace/results/stage3c/stage3c_review_packet.md`
- 机器可读：`E0_stats.json`、`stage3c_promotion.json`
