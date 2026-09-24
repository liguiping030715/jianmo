# Q1-FINAL 方法：完整特征提取与时间对齐

> 仅使用**附件1**（100 条 MOSEI 原始 mp4）。附件2/3/4 不作为训练数据、对齐真值或特征目标。
> 目标：为每条原始样本建立**经过验证的统一时间轴**，并在其上提取文本、音频、视觉特征。

## 1. 原始数据接口
- 输入：`附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/<video_id>/<clip>.mp4`，
  共 100 个片段、37 个 YouTube-ID 文件夹；每个 mp4 内为 H.264 视频 + AAC 音频。
- 官方转录与标签来自清洗清单 `attachment1_manifest.json`（字段 id / text / label / annotation / raw_video）。
- 不删除任何样本；提取失败的模态记录为显式观察状态。

## 2. 文本特征提取
- 模型：`bert-base-uncased`（本地、冻结），Tokenizer 为 fast 版本。
- 对转录编码（含 offset mapping、word_ids），BERT 前向取最后一层隐状态；
  同一 word 的子词状态取均值，得到 **768 维上下文词嵌入**，并记录每词字符跨度。
- **内容存在与时间对齐分离**：100 条官方转录均保留；`text_word_feat`、字符跨度、
  `text_word_surface` 和 `text_utterance_feat` 为样本级内容输出，含全部 22 条时间未解析样本。
  原始提取器仅保存了 50-bin 投影，语义补丁因此只用相同冻结 BERT 和官方转录补算
  词级内容；没有重跑视频、CTC、openSMILE 或 FaceMesh。
- 不将单词在视频时长上均匀分布。

## 3. 音频特征提取
- 用 PyAV 将 AAC 音频确定性重采样为 **16 kHz、单声道、s16 PCM**。
- 特征提取器：openSMILE **eGeMAPSv02 Low-Level-Descriptors**（**25 维 LLD**），
  hop≈0.01 s、分析窗≈0.02 s；每个 LLD 帧记录精确的片段相对起止时间。
- 不静默替换提取器：openSMILE 不可用时报告 blocked，而非改用其他特征。

## 4. 视觉特征提取
- 视频帧来自验证后的呈现时间轴（解码 PTS），逐帧以 static 模式运行
  MediaPipe **FaceDetection（model_selection=1, min_conf=0.5）+ FaceMesh（468 关键点）**。
- 每帧记录：解码帧序号、呈现时间戳、检测状态、**468×3=1404 维**几何特征、可用时的置信度。
- 检测不到人脸的帧**保留**，记为“人脸不可观测”，绝不删帧。

## 5. 验证时间戳构造
- 使用 PyAV 单遍 `demux()`，按 packet 所属流分发到视频/音频解码器（避免顺序解码导致另一流为空）。
- 视频帧时间 = 解码 **PTS × time_base**；音频时间 = 实际样本/包时间线。
- 公共时间格时长 D 取 PyAV/FFmpeg 给出的**编辑后有效呈现时长**；
  原始 `mvhd` 时长另行记录，不把它当成可播放时长。视频/音频实际覆盖末端单独记录用于 QA。
- **不使用 frame_index / nominal FPS**。核查发现旧清单用“元数据帧数÷标称帧率”系统性高估时长
  （平均相对高估约 0.293，92/100 样本），例如某片段旧值 8.767 s，实际仅 5.514 s。

## 6. 文本/音频强制对齐
- 采用通用（**非情感训练**）语音模型 `wav2vec2-base-960h` 的 CTC 发射概率，
  通过 torchaudio `forced_align`（成熟 Viterbi）+ `merge_tokens` 将转录对齐到音频。
- CTC 帧步长 0.02 s；目标序列为大写字符、词间以 `|` 边界。
- 由于 CTC 总会经过所有目标 token，**以置信度判定**：词置信度（词内 token 概率均值）
  ≥ **0.30** 才接受其时间区间；标点/纯数字等无语音实体的 token 排除。
- 样本在“可对齐词”中接受比例 ≥ **0.80** 标记 VERIFIED，否则 UNRESOLVED；
  对齐失败不编造时间戳。

## 7. 50-bin 数学对齐
- 对时长 D 的样本建立 K=50 个公共时间 bin：B_k=[kD/K,(k+1)D/K)，k=0,…,49。
- **文本**：对词 j 的区间 W_j 与 bin 交叠时长 ω_jk 加权聚合词嵌入
  T_k = Σ_j ω_jk h_j / Σ_j ω_jk（分母非零时）。
- **音频**：LLD 帧按其区间与 bin 的交叠时长加权平均。
- **视觉**：帧按 PTS 落入 bin，同 bin 检测成功帧的关键点取均值。
- 绝不依据 token 序号推断文本秒数。
- 每个 bin 的源映射在 `source_maps/<id>.json`：已验证词 ID、音频 LLD 帧 ID/时间区间、
  解码视频帧 ID/PTS 候选集。原提取器未保存逐帧 FaceMesh 结果，故视觉候选帧
  与真正贡献关键点均值的帧不能全部一一证实；见 `q1_traceability_review.md`。

## 8. 内容、对齐与 P/O 状态（每模态/bin 分离存储）
- **CONTENT EXISTS**：100/100 样本均有官方转录与 BERT 词级/话语级内容，
  `text_content_present=1`。该状态不含时间定位主张。
- **TEMPORAL LOCATION VERIFIED**：78/100 样本 `text_alignment_state=1`；
  经接受的词时间区间可以用于 50-bin 投影。
- **TEMPORAL LOCATION UNRESOLVED**：22/100 样本 `text_alignment_state=2`；
  文本内容和 BERT 输出仍在，但不声称任何词处于特定时间 bin。
  `text_alignment_state=0` 预留给经验证不存在/不适用，本批次没有此状态。
- 文本 bin 的 **P/O 取值 0/1/2**：0=有独立证据证明文本不存在/不适用，
  1=已验证词覆盖/有可用的局部嵌入，2=时间位置未解析。
  本批次没有足以证明文本局部不存在的证据，因此文本 bin 实际只取 1 或 2。
  22 条整样本未解析者全部 50 格为 `P=2,O=2`；78 条整样本已验证者
  仍有低置信词和未覆盖时间格，后者也取 2，**不是“无文本”或已验证 padding**。
- 音频/视觉原有 P/O 为二值：P=验证的媒体时间/解码帧存在；O=该 bin
  有有限的提取器观测。视觉有帧但无人脸时 P=1/O=0。
- 已验证观测恒有 O=1 ⇒ P=1；**绝不从零向量推断缺失**。
  `*_feat` 保留原提取数组（未知处可为 NaN）；`*_model_input` 在 O≠1 处
  置零作为有限数值占位，必须与 P/O 和对齐状态共同输入，零本身无缺失语义。

## 9. 质量控制
- 先做确定性五样本 smoke（最短/中位/最长 + 标签多样，不按提取质量挑），通过后冻结全部提取器。
- 全量 100 样本审计：源文件存在、转录/标签配对、验证时长、解码成功、恰 50 bin、
  无 NaN/Inf、文本对齐覆盖、音频覆盖、视觉人脸覆盖、P/O 一致性、可复现 hash。
- 结果：100/100 源文件存在、已观测特征与模型数值输入 finite、P/O 一致；
  文本强制对齐 78/100 VERIFIED、22/100 UNRESOLVED，转录和 BERT 内容 100/100。
  已验证 bin 覆盖（O=1，中位/50）：文本 35、音频 50、视觉 50；
  22 条文本时间未解析样本的 O=2，不计为已验证缺失。

## 10. 局限
- 22/100 样本文本对齐 UNRESOLVED：经核查主要为音频非英语（转录为英语翻译）、
  静音/纯音乐、或音频与转录不一致；这里只能断言**时间定位未被验证**，
  不把其转录或 BERT 内容判为缺失，也不编造词时间戳。
- 即使整条样本标为 VERIFIED，个别词仍可能因置信度低而时间未解析；
  图中样本的 `who` 和第二个 `or` 即为实例。文本 O=1 表示至少一个词可用，
  不表示整格词项覆盖完整。
- 10/100 样本视觉无人脸观测（远景/侧脸/遮挡），对应 bin O=0。
- wav2vec2-base-960h 训练于 LibriSpeech，对口音、强背景噪声、重叠语音的对齐置信度偏低；
  阈值（0.30 / 0.80）为经验设定，边界样本可能波动。
- 特征为冻结的预训练模型输出，未针对 MOSEI 情感任务微调。

## 11. 复现设置
- 提取器名称、版本、权重 SHA256、特征定义、窗/跳长、50-bin 规则、对齐配置见
  `q1_extractor_freeze.json` / `.md`。
- 冻结配置中旧写法 `duration_authority=container mvhd (verified)` 是描述错误；
  实际 `q1_media.py` 用 `av.open(...).duration`，即有效呈现时长。源容器 `mvhd`
  仍可长于该值，具体差异见逐条 source map。冻结文件保持原哈希，仅在此更正解释。
- 每样本输出：`features/<id>.npz`（三模态时间特征与 P/O、词级及话语级
  BERT 内容、对齐状态、数值安全接口）+ `features/<id>.json`（provenance）。
- 清单与报告：`q1_manifest.csv`、`q1_alignment_report.csv`、`q1_quality_report.md`；
  原典型样本图：`figures/typical_alignment.png`（仅展示一条 VERIFIED 样本，
  且未画低置信度词）；修订图 `figures/typical_alignment_audited.png` 明列
  未对齐词、音频尾部和 P/O。完整证据索引见 `q1_traceability_review.md`。
- 边界：附件1 结果只用于问题1，不并入 Q2 训练，不访问 A2 test/A3/A4，不自动恢复 Q2。
