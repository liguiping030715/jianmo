# P3 reusable_patterns: 信号/图像/空间数据模式（带适用条件）

> P1.5 据 2024 E/F + 2025 C/E 深读重写。

---

## Pattern S-1：视频/图像检测跟踪
- **Applicable When**：需从视频/图像中定位并持续跟踪目标对象。
- **Why It Works**：检测器（YOLO 类）定位 + 跟踪器（SORT/ByteTrack 类）维持时序 ID，是成熟两件套。
- **Evidence**：2024E YOLOv8 检测跟踪 + 透视变换校正近大远小。
- **Failure Modes**：透视依赖人工选点；遮挡/密集目标丢 ID；视角畸变未校正致计数偏差。
- **Do Not Use When**：目标无需跨帧关联时，只用检测即可，不必加跟踪。

## Pattern S-2：强噪声信号先分解降噪
- **Applicable When**：信号非平稳、强噪声、目标成分被淹没。
- **Why It Works**：EMD 类（CEEMDAN/TVFEMD）分解成 IMF，按包络谱峭度等选有效分量再处理，优于直接喂原始信号。
- **Evidence**：2025E 手工+时频特征、DWT 小波图；2024F 信号时序分析。
- **Failure Modes**：分解模态混叠；IMF 选择误删有用成分。
- **Do Not Use When**：信号本身平稳、信噪比高时无需分解，过度处理反失真。

## Pattern S-3：多域特征联合
- **Applicable When**：单一域特征判别力不足。
- **Why It Works**：时域（快速评估）+频域/包络（定位类型）+时频域（非平稳演化）信息互补。
- **Evidence**：2025E 时域/频域/时频/包络四维 + ViT 深度特征融合；2024C HHT+双谱多域融合。
- **Failure Modes**：特征堆砌、维度高样本小导致过拟合；深度特征缺可解释。
- **Do Not Use When**：样本量不足以支撑高维特征时，应先降维/筛选。

## Pattern S-4：空间数据统一分辨率 + 趋势/相关检验
- **Applicable When**：多源栅格/地理数据时空分辨率与坐标系不一致。
- **Why It Works**：重采样统一分辨率、时间对齐；MK 检验+Sen 斜率（抗异常）判趋势；Moran's I 判空间自相关；各向异性距离处理方向差异。
- **Evidence**：2024D MK+Sen、DEM 水文分析；2025D 各向异性距离（γ≈10）。
- **Failure Modes**：线性外推假设趋势延续；以相关+物理解释替代因果（2024D R²仅0.1–0.56 却结论偏强）。
- **Do Not Use When**：趋势不显著时不应外推（应保持当前值）；各向同性距离在强各向异性场失效。
