# Stage 3B Review Packet — 2026E

## A. 范围与冻结

按两步走执行：先 Stage 4.5 定向复核（不重新清洗），通过后才进入 D0/D1。
- 数据/接口完全复用已验证的 pilot_stage4.py：aligned v2、clean/gap BERT cache、mask manifest、
  augmented 输入、metrics、selection loss；pilot_stage3b 仅替换模型前向与损失包装，无第二套管线。
- 未重训/重调 C1、C2、RU；未改阈值；A2 test/A3/A4 不参与选择。
- 损失口径 CE + λ·Huber，λ=2.0526，AdamW lr1e-3 wd1e-4，epoch cap12/patience3。

## B. Stage 4.5 定向复核

状态：**PREPROCESSING_REAUDIT_PASS**
- 脚本独立检查：aligned/unaligned 计数与 ID 唯一、aligned↔unaligned 标签一一对应、
  scaler train-only 且 hash 一致、run config 无 A3/A4/test 引用。
- 经现有 loader 执行的 8 条语义断言（A1–A8）全部 True：
  注入位置原始 P=1,O=1；注入后仅 injected_mask 上 P=1,O=0；native=2 注入外不变；
  R0a/R1 共用同一 mask manifest；文本 gap 先改 token 再走 BERT、key 由修改后 token 决定；
  clean BERT state 不复用；scaler/数据跨 run 一致；samples→batch→prediction 顺序与标签保持。
- 结论：排除预处理/接口实现问题，Stage 4 的失败属真实模型失败。

## C. D0 / H8

| seed | Δ macro-F1 | Δ MAE |
|---|---|---|
| 1729 | -0.0161 | +0.0157 |
| 2718 | -0.0392 | -0.0062 |
| 31415 | +0.0532 | -0.0117 |
| **均值** | **-0.0007** | **-0.0007** |

- guard（ΔF1≥−0.01、ΔMAE≤+0.02）：通过；
- 实际改善门槛（任一任务 +0.015）：**不通过**（ΔF1 -0.0007、ΔMAE -0.0007）；
- **D0 = NOT_PROMOTED，H8 NOT_SUPPORTED**：任务特定融合相对 shared fusion 无可测收益。

## D. D1 / H9

- 仅完成 seed1729 机制 smoke：代码可运行，local scorer 对注入区间正确响应（w_gap→0、边界过渡、
  unknown 保留小权重），非静态位置模板。
- D0 未通过 → 不跑 D1 全三 seed，**H9 未判定**。

## E. 决策汇总

| 项 | 结论 |
|---|---|
| Stage 4.5 复核 | PREPROCESSING_REAUDIT_PASS |
| D0（任务特定融合） | NOT_PROMOTED / H8 NOT_SUPPORTED |
| D1（局部状态池化） | MECHANISM_SMOKE_ONLY / H9 未判定 |
| C1、C2、RU | 维持 Stage 4 冻结决策，不重跑 |

## F. 结论

Stage 3B 未产生任何通过预注册门槛的候选；shared→task-specific fusion 与提高时间分辨率均未带来可测收益。
不进入全量实验，不自动重设计模型。

STAGE3B_NO_PROMOTED_CANDIDATE
