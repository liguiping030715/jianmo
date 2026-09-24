# Model Card — E0 (Local State-Aware Temporal Pooling)

## 身份与定位
- 名称：E0；Stage 3C 独立 H9 测试。
- 基线：frozen R1/B1；直接对照 frozen R1。
- 唯一架构差异：content temporal pooling 由 masked-mean 换为 Conv1d(kernel=3) local scorer
  + masked-softmax 加权；state summary 保留。
- 保留 R1 全部：modality encoders、P/O tri-state 接口、shared fusion、cls/reg heads、
  CE + λ·Huber、AdamW、λ、train/valid 协议。
- 不含：D0 task-specific fusion、全局模态 gate、Transformer/MoE/distillation/reconstruction。

## 状态规则
P=0 硬排除；O=0 排除 content；P/O=2 以显式 unknown 保留；零幅值不定义缺失。

## 训练配置（冻结）
λ=2.0526；AdamW lr1e-3 wd1e-4；epoch cap12/patience3；
seeds 1729/2718/31415；aligned_50；train/valid 3395/728。

## 结果（clean，三 seed）
| seed | Δ macro-F1 | Δ MAE | 分类 |
|---|---|---|---|
| 1729 | -0.1191 | +0.0068 | D |
| 2718 | -0.0015 | -0.0202 | B |
| 31415 | +0.0118 | -0.0261 | A |
| **均值** | **-0.0363** | **-0.0132** | — |

## 机制
local scorer 对注入区间正确响应（w_gap=0、边界过渡），非静态位置模板，unknown 保留。

## 结论
**E0_REJECTED。**
- clean F1 guard 失败（-0.0363 < −0.015）：seed1729 F1 崩溃
  （-0.1191）拖累均值；
- gap 无达标增益（F1 gain -0.0331、MAE red +0.0098
  < 0.015）；
- A/B/C/D = D/B/A，方向不稳定，不构成稳定任务依赖 hypothesis。

## 观察与局限（如实记录）
- 回归 MAE 在 2/3 seed 改善（2718 -0.0202、
  31415 -0.0261），5/6 scenario MAE 正向；seed31415 双任务改善。
- 但 seed1729 分类 F1 崩溃，表明该结构训练高方差、对初始化/训练轨迹敏感；
  local weights 是机制诊断而非解释，不据此晋级。
- best epoch：{1729: 2, 2718: 2, 31415: 2}；训练墙钟（秒）：{1729: 46.6, 2718: 53.1, 31415: 53.1}。
