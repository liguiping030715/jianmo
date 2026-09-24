
# Stage 4.5 + Stage 3B 实验方案

## 0. 目标

不要重新清洗。先用一次定向 re-audit 排除“样本错位、scaler 泄漏、mask 注入错误、文本 gap 复用 clean BERT state”等实现问题。

只有 `PREPROCESSING_REAUDIT_PASS` 后，才进入 D0/D1。

---

## 1. Stage 4.5 — targeted preprocessing re-audit

必须通过以下硬检查：

1. `aligned/unaligned` train/valid ID、标签一一对应。
2. `scalers.json` 仍为 train-only，hash 与 Stage 4 使用的一致。
3. raw/v2/sample order -> batch -> prediction ID 完整一致。
4. synthetic gap 只允许注入原始 `P=1,O=1` 的连续位置。
5. 注入后只在 `injected_mask` 上改为 `P=1,O=0`。
6. 原生 `P/O=2` 在非注入位置不被改写。
7. R0a/R1/D0/D1 对相同 seed/scenario/sample 使用相同 aligned mask manifest。
8. 文本 gap 必须先改 token ID，再走 frozen BERT；cache key 必须由修改后的 token IDs 决定。
9. 不能给 text-gap 样本复用 clean contextual BERT states。
10. D0/D1 使用与 Stage 4 完全相同的 scaler、标签、split 与 frozen BERT。

若任一失败：停止，修实现；不要进入模型重设计。

---

## 2. D0 — Task-Specific Fusion

**直接对照：R1/B1 -> D0**

除 fusion 外保持 R1 的接口和轻量编码/池化逻辑不变。

R1:
`[hT,hA,hV] -> shared fusion -> cls/reg heads`

D0:
`[hT,hA,hV] -> F_cls -> cls head`
`[hT,hA,hV] -> F_reg -> reg head`

科学问题：

> shared fusion 是否造成 classification / regression 的负迁移？

不要加入：
- reliability gate
- Transformer
- MoE
- distillation
- reconstruction

### D0 必查

- clean macro-F1 / MAE / Pearson
- S1-S6 gap F1 / MAE
- 三 seed 方差
- 每任务的 modality ablation
- 可选：classification loss 与 regression loss 对共享 encoder 参数的 gradient cosine，仅作冲突诊断，不作解释

---

## 3. D1 — Local State-Aware Temporal Pooling

**直接对照：D0 -> D1**

只把 mean pooling 替换为 position/local-window state-aware pooling。

对位置 t：

`z_mt = encoder(x_mt, P_mt, O_mt, pos_t)`

浅层局部 scorer（例如 kernel=3 Conv1d）得到：

`score_mt`

eligibility：

- `P=0`：排除
- `O=0`：已知 synthetic unavailable，排除 content
- `P/O=2`：保留，作为 unknown state 进入 scorer
- zero magnitude：绝不定义 missing

`w_mt = masked_softmax(score_mt)`

`h_m = sum_t w_mt z_mt + state_summary`

科学问题：

> 局部连续缺失是否应在 pooling 前以 position-level 状态处理，而不是 pooling 后给整个模态一个全局权重？

### D1 必查

- gap 内平均权重是否下降
- gap 边界两侧权重变化
- unknown 位置平均权重
- position-wise mean weight 是否固定成静态位置模板
- local-weight entropy
- 不同 seed/scenario 的局部权重稳定性

**禁止把 local weight 写成 Q3 explanation。**

---

## 4. 实验矩阵

保持 Stage 4 已冻结的：
- aligned_50
- train=3395 / valid=728
- seeds = 1729, 2718, 31415
- S1-S6
- frozen BERT
- train-only scaler
- CE + Huber
- 同一 lambda
- 同一 optimizer / LR search budget
- 同一 epoch cap / patience
- A2 test / A3 / A4 不参与选择

对照表：

| run | change | direct comparator |
|---|---|---|
| frozen R0a | augmentation reference | — |
| frozen R1 | tri-state B1 | R0a |
| D0 | task-specific fusion | **R1** |
| D1 | D0 + local temporal pooling | **D0** |

不要重跑/重调 C1、C2、RU。

---

## 5. 新的预注册筛选规则

这些阈值必须在 D0/D1 结果产生前冻结。

### D0 gate

D0 相对 frozen R1：

- 三 seed 平均 clean macro-F1 不低于 R1 超过 0.01；
- 三 seed 平均 clean MAE 不高于 R1 超过 0.02；
- 至少一个任务有实际改善：
  - macro-F1 `>= +0.015`，或
  - MAE `<= -0.015`；
- 另一任务不得出现明显退化；
- 改善方向至少 2/3 seeds 一致；
- 若一个任务提升完全以另一个任务明显下降为代价，则 H8 不成立。

D0 未通过：停止，不运行 D1 全三 seed；最多允许 first-seed mechanism smoke。

### D1 gate

D1 相对 D0：

- clean guard：F1 不低于 0.015，MAE 不高于 0.025；
- 六情景平均 gap macro-F1 `>= +0.010` **或** gap MAE `<= -0.015`；
- 同时对应 clean-to-gap degradation 有改善；
- 改善方向至少 2/3 seeds、4/6 scenarios；
- local weights 必须对 injected interval 有响应；
- 若 local weights 主要固定成 position template、或与 gap 无关，则机制判失败；
- unknown-retained 仍为主策略，不能靠硬删 unknown 获得晋级。

说明：这是 Stage 3B screening gate，不是最终论文显著性结论。

---

## 6. 执行顺序

1. `stage4_5_reaudit.py`
2. 把 8 个 existing-loader assertions 接入现有 `pilot_stage4.py` 并跑 train+valid
3. 冻结 re-audit JSON
4. 只跑 D0 seed1729
5. 预算/实现正常后跑 D0 seeds 2718/31415
6. Reviewer 判断 D0 gate
7. 只有 D0 有效时，跑 D1 seed1729
8. 若 D1 first seed 非灾难性失败，再补另外两 seed
9. STOP，生成 Stage3B review packet
10. 不进入 A2 test/A3/A4

---

## 7. 结果文件

建议：

`workspace/results/stage3b/preprocessing_reaudit.json`

`workspace/results/stage3b/D0_comparison.md`

`workspace/results/stage3b/D1_comparison.md`

`workspace/results/stage3b/stage3b_review_packet.md`

机器可读：

`comparison_statistics.json`

`promotion_decisions.json`

若无候选通过：

`STAGE3B_NO_PROMOTED_CANDIDATE`

若有：

`STAGE3B_HAS_PROMOTED_CANDIDATE`
