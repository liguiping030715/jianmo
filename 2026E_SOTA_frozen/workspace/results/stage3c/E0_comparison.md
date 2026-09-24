# E0 对比 — Local State-Aware Temporal Pooling（独立 H9 测试）

E0 = frozen R1/B1 + 仅替换 content temporal pooling（masked-mean → Conv1d(kernel=3) local scorer
+ masked-softmax 加权；state summary、shared fusion、heads、losses、optimizer、λ、协议全部不变）。
R1 vs E0 仅 temporal pooling 不同；不使用 D0 task-specific fusion。

## 1. Clean valid（三 seed）

| 候选 | seed | Accuracy | macro-F1 | MAE | Pearson |
|---|---|---|---|---|---|
| R1 | 1729 | 0.6360 | 0.6153 | 0.6038 | 0.6351 |
| R1 | 2718 | 0.6099 | 0.5572 | 0.6043 | 0.6331 |
| R1 | 31415 | 0.6209 | 0.5449 | 0.6134 | 0.6420 |
| E0 | 1729 | 0.6058 | 0.4963 | 0.6106 | 0.6719 |
| E0 | 2718 | 0.6209 | 0.5557 | 0.5840 | 0.6765 |
| E0 | 31415 | 0.6319 | 0.5567 | 0.5873 | 0.6717 |
| **R1 均值** | — | **0.6223** | **0.5725** | **0.6072** | **0.6367** |
| **E0 均值** | — | **0.6195** | **0.5362** | **0.5940** | **0.6733** |

## 2. E0 − R1 与 A/B/C/D

| seed | Δ macro-F1 | Δ MAE | 分类 |
|---|---|---|---|
| 1729 | -0.1191 | +0.0068 | D |
| 2718 | -0.0015 | -0.0202 | B |
| 31415 | +0.0118 | -0.0261 | A |
| **均值** | **-0.0363** | **-0.0132** | — |

A=两任务都改善；B=回归改善分类退化；C=分类改善回归退化；D=都不改善。

## 3. Gap 增益（共同 injectable，三 seed 均值）

| scenario | gap F1 gain | gap MAE red |
|---|---|---|
| S1 | -0.0338 | +0.0128 |
| S2 | -0.0370 | +0.0158 |
| S3 | -0.0309 | +0.0155 |
| S4 | -0.0374 | +0.0037 |
| S5 | -0.0307 | -0.0040 |
| S6 | -0.0292 | +0.0150 |
| **均值** | **-0.0331** | **+0.0098** |

- 有用效应门槛：gap F1 gain ≥ +0.010 或 gap MAE red ≥ +0.015；
- 实际：F1 gain -0.0331、MAE red +0.0098，
  方向 scenario：F1 正向 0/6、MAE 正向 5/6。

## 4. Local 机制（验证，非 explanation）

| modality | w_gap | w_boundary | w_unknown | entropy |
|---|---|---|---|---|
| text | 0.000 | 0.069 | n/a | 2.020 |
| audio | 0.000 | 0.047 | 0.005 | 3.357 |
| vision | 0.000 | 0.064 | 0.003 | 2.950 |

- 含 gap 模态注入区间内平均权重 w_gap=0.000，边界 w_boundary 有过渡（0.05–0.07），
  机制对注入缺失有响应；
- position 权重跨 scenario 的标准差 text/audio/vision ≈
  0.0033/
  0.0038/
  0.0041，非静态位置模板；
- unknown 位保留小权重（audio/vision），aligned text 无 unknown=2。

## 5. 判定

- Clean guards：ΔF1 ≥ −0.015 → **False**（-0.0363）；
  ΔMAE ≤ +0.025 → **True**（-0.0132）。
- 有用效应：**False**。
- A/B/C/D：D/B/A，B 仅 1 seed、C 0 seed，不满足稳定 B/C（≥2 seed）。

**E0_REJECTED。**
