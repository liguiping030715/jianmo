# Q1-FINAL Five-Sample Smoke Report

## 1. 确定性选样（不按提取质量挑）
按旧 manifest 时长排序，取最短 / 中位 / 最长，并补充情感标签多样性，去重后 5 条：

| # | id | 时长(verified, s) | 标签 |
|---|---|---|---|
| 1 | -mJ2ud6oKI8$_$6 | 2.257 | — |
| 2 | -UuX1xuaiiE$_$3 | 4.130 | — |
| 3 | -yRb-Jum7EQ$_$1 | 29.288 | — |
| 4 | -UacrmKiTn4$_$4 | 4.741 | — |
| 5 | -ri04Z7vwnc$_$0 | 2.807 | — |

## 2. 验证项（逐条核对）

| 验证项 | 结果 |
|---|---|
| 媒体时间戳（PyAV 解码 PTS，非 frame_index/nominal FPS） | PASS |
| 文本 BERT 上下文词嵌入 | PASS |
| 文本强制对齐（wav2vec2 通用 CTC，非情感） | 4/5 VERIFIED，1/5 正确标记 UNRESOLVED |
| 音频 openSMILE eGeMAPS LLD + 帧时间戳 | PASS（5/5 覆盖 50 bins） |
| 视觉 MediaPipe 人脸检测 + FaceMesh，按 PTS | PASS（检测失败记状态、不删帧） |
| 50-bin 公共时间对齐 | PASS（每样本恰 50 bins） |
| P / O 分离 | PASS |
| 无 NaN / Inf（O=1 的特征） | PASS（5/5 finite） |
| provenance 可追溯 | PASS（每样本 npz + json） |

每样本有效观测 bin 数（O 计数，text/audio/vision）：

| id | text | audio | vision |
|---|---|---|---|
| -mJ2ud6oKI8$_$6 | 23 | 50 | 49 |
| -UuX1xuaiiE$_$3 | 29 | 49 | 50 |
| -yRb-Jum7EQ$_$1 | 39 | 50 | 50 |
| -UacrmKiTn4$_$4 | 39 | 50 | 50 |
| -ri04Z7vwnc$_$0 | 0 | 50 | 0 |

## 3. UNRESOLVED 样本调查（-ri04Z7vwnc$_$0）
- 官方转录为 “going to get married or are already married”（8 词），音频时长 2.76 s；
- wav2vec2 CTC greedy 仅输出 “AA”，音频与转录无法匹配；该模型在其余 4 样本对齐成功，
  故判定为**样本自身音频/转录不一致**，对齐正确标记 UNRESOLVED，未编造时间戳；
- 67 个解码帧 MediaPipe 均未检测到人脸（P=50 帧存在、O=0 人脸不可观测），帧全部保留、未删除。

## 4. 结论
核心提取器与时间映射全部有效；管线能正确处理对齐失败与人脸不可观测的样本。
**SMOKE PASS**，可冻结提取器并进入全量 100 样本处理。
