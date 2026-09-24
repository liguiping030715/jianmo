# D1 — Local State-Aware Temporal Pooling（first-seed 机制 smoke）

**这是 MECHANISM SMOKE，不是 H9 判定。** 按预注册规则，D0 gate 未通过时不运行 D1 全三 seed；
此处仅跑 D1 seed=1729 以验证局部状态感知池化代码可运行、机制路径正确。

## 1. Clean（仅 seed1729，不与门槛比较）

| 候选 | Accuracy | macro-F1 | MAE | Pearson |
|---|---|---|---|---|
| D0 | 0.6401 | 0.5992 | 0.6195 | 0.6347 |
| D1(smoke) | 0.6319 | 0.5679 | 0.5934 | 0.6575 |

单 seed 数字不构成 H9 证据，不做方向或显著性结论。

## 2. 局部权重机制（验证代码路径）

| scenario / modality | w_gap | w_boundary | w_unknown | entropy |
|---|---|---|---|---|
| S1/text | 0.000 | 0.028 | nan | 2.142 |
| S1/audio | nan | nan | 0.002 | 3.002 |
| S1/vision | nan | nan | 0.009 | 3.641 |
| S4/text | 0.000 | 0.034 | nan | 2.090 |
| S4/audio | 0.000 | 0.042 | 0.002 | 2.800 |
| S4/vision | nan | nan | 0.009 | 3.641 |
| S6/text | nan | nan | nan | 2.256 |
| S6/audio | 0.000 | 0.053 | 0.002 | 2.900 |
| S6/vision | 0.000 | 0.042 | 0.010 | 3.610 |

机制观察（仅 smoke）：
- 含 gap 的模态，注入区间内平均权重 w_gap 被压到 0.0，边界 w_boundary 有小幅过渡（约 0.03–0.05），
  说明 Conv1d scorer 对注入缺失有响应，而非恒定静态位置模板；
- unknown 位置保留了很小权重（audio/vision 约 0.002–0.010），未被硬删（aligned text 无 unknown=2，故 nan）；
- vision 权重熵较高（≈3.6，接近均匀），text 较集中（≈2.1）。

## 3. 处置

D0 未通过 → 不补跑 D1 其余两 seed、不判 H9。D1 实现已验证可运行，留待后续假设需要时复用。
