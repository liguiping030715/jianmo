# P3 reusable_patterns: 信号/图像/空间数据通用

> 从 2024 E/F + 2025 C/E 提炼。

## 1. 视频/图像题
- 检测+跟踪两件套（YOLO + DeepSORT/ByteTrack）。
- 预处理：透视变换校正视角、CLAHE、双边/中值滤波。
- 检测模型提分：注意力(SE/CBAM) + 改进backbone + 残差防过拟合。
- 二分割类别不平衡 → BCE+Dice复合损失，报 IoU/Recall不只Accuracy。

## 2. 信号题
- 强噪声先 EMD类分解(TVFEMD/CEEMDAN) 选IMF再降噪，别直接喂原始信号。
- IMF选择用包络谱峭度/WESK。
- 特征四维度：时域/频域/时频域/包络谱。

## 3. 空间/地理数据
- 多源栅格先重采样统一分辨率+时间对齐。
- 时序趋势：MK检验+Sen斜率+时序三分解。
- 空间自相关 Moran's I。
- 各向异性距离（垂直方向加γ惩罚）。
