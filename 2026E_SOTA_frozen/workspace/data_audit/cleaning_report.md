# 2026E 最小必要清洗执行报告

**CLEANING_READY_WITH_LIMITATIONS** — 已按预先冻结的 [cleaning_contract.md](../evidence_recovery/cleaning_contract.md) 执行规范化、掩码化与训练集内标准化；独立核查全部通过。本状态只表示**清洗产物可供审查**，不表示 Stage 2.5 已解除阻塞或获准进入 Pilot。

有效产物：[`workspace/data/processed/cleaning_2026e_v2/`](../data/processed/cleaning_2026e_v2/)。版本 `cleaning_2026e_v1` 在首次运行时因专项文件目录识别条件错误而中止，已写入 `INCOMPLETE_DO_NOT_USE.json`，**不得使用**。v2 从原始数据重新完整生成，没有沿用 v1 输出。

## 范围与规则

原始数据始终只读。两套 Attachment 2 特征版本均处理，未提前选定 aligned/unaligned。训练、验证、测试分别为 **3,395 / 728 / 727** 条；附件3每版 **30** 条，附件4每版 **20** 条；附件1保留 **100** 条原始视频与标签的一对一清单。没有删样本、改标签、插补零段、平滑专项输入或按 `vision_lengths` 截断视觉序列。除训练/验证原标签的核对副本外，处理目录不输出测试集或专项集标签。

`P/O` 使用三值编码：`0=已知填充`、`1=可信位置/可用观测`、`2=未知`。额外的 `zero_context` 使用 `0=非零`、`1=已知填充`、`2=原生零或无法判别`、`3=专项集内部疑似零缺口`。**代码 3 不是人工缺失真值。**可靠性 `R` 尚无官方观测字段，留待模型估计；清洗阶段没有伪造数值。

## 变换台账

| 输入字段 | 执行规则与原因 | 拟合划分 / 参数 | 影响划分 | 可逆性与信息损失 | 泄漏风险控制 |
|---|---|---|---|---|---|
| `audio`, `vision` 的 float64/float32 | 先检查有限值和 float32 精确往返，再统一存为 float32，简化接口和存储 | 无拟合；所有源值必须精确可表示 | Attachment 2 train/valid/test，附件3/4 | 原始数值转换精确；原始文件不变 | 任何不精确值令程序失败，不能静默量化。 |
| aligned 附件3 `text_bert` | 检查有限、整数、形状、0/1 注意力与后缀填充后转 int64 | 无拟合；30/30 通过精确往返 | 附件3 aligned | 精确可逆；原始 float32 保留于 raw | 不用专项结果拟合或选编码器。 |
| `id`, `raw_text`, 本地文件名 | 原字符串保留，另建样本元数据；附件3无 ID，明确记录 `null` 和本地文件键 | 无拟合；保持原始排序 | Attachment 2/3/4；附件1一对一清单 | 可逆，无重编号；不生成伪 CMU ID | 核对 ID 与源文件，避免文件名误当原始 ID。 |
| `text_bert` 注意力行 | 构造文本 `P/O`；连续文本特征的零值不当作 padding | 无拟合；要求二值、连续后缀填充 | 有 token 字段的全部划分 | 原特征不改，只增掩码 | train/valid/test 同一确定规则。 |
| unaligned `audio_lengths` | 仅在与非零前缀完全一致时构造 `P/O` | 无拟合；逐样本一致性门 | Attachment 2/4 unaligned | 原特征不改；原长度另存 | 附件3无长度，不能借用此规则推断末端。 |
| aligned 音视频和 unaligned 视觉 | 非零向量标记为可信观测；零向量保留且 `P/O=未知`；`vision_lengths` 原值另存、不截断 | 无拟合；三值掩码 | Attachment 2/3/4 对应字段 | 无删减；不把零变成缺失标签 | 不依据专项集结果确定掩码或阈值。 |
| 附件3音视频零段 | 记录内部零段为 `zero_context=3` 候选，其 `P/O` 仍为未知；边界零段维持未知 | 无拟合；固定的非零包络描述规则 | 仅附件3的诊断元数据 | 不改特征、不插补；候选解释有不确定性 | 不把观察到的专项模式当训练用缺口分布。 |
| `text`, `audio`, `vision` 数值特征 | 逐通道 z-score；只对可信位置执行；原有全零向量完整保留 | **仅 Attachment 2 train 拟合**每版每模态 `μ,σ`；零方差通道规则 `σ=1` | 以固定参数应用于 train/valid/test 和附件3/4实际存在的同名数值字段 | 保存 `μ,σ` 后近似可逆；标准化有浮点舍入；raw 不变 | 不对 valid/test/专项数据拟合；不在缺少数值 `text` 的附件3伪造同名字段。 |
| Attachment 1 视频、标签 | 只生成 100 行来源清单和容器级时长，不抽样删除或重编码 | 无拟合 | Attachment 1 | 视频与标签保持原样 | 与 Q2/Q3 训练特征隔离。 |

各版本 `scalers.json` 保存逐通道 `μ/σ`、拟合划分和可信行数。训练集可信行数：aligned 文本 **83,672**、语音 **76,817**、视觉 **72,633**；unaligned 文本 **83,672**、语音 **500,044**、视觉 **353,483**。对应训练集标准化后各通道均值绝对值的最大值接近 0，标准差为 1（浮点精度内）。valid/test 的标准差无需等于 1；它们从未参与拟合。详细统计见 [cleaning_statistics.json](../results/data_audit/cleaning_statistics.json)。

## 独立 sanity check

[cleaning_verification.json](../results/data_audit/cleaning_verification.json) 报告 `passed=true`、**0 项错误**。独立复核内容：

- 处理前后形状和条数一致；Attachment 2 ID 原样、train/valid 标签逐项相等，测试/专项标签不输出；附件1清单 100 个唯一 ID 和 100 个存在的视频。
- 所有处理数组无 NaN/Inf；`P/O/zero_context` 只含声明的状态值。
- 音视频每个样本的全零向量位置完全一致，因此零段数量、位置、长度也保持一致；没有把附件3的缺失输入补掉。
- 对各划分抽查标准化公式，均使用相同版本的 train 参数；附件3对齐版 token dtype 转换与原值精确一致。
- 244 个处理涉及的 PKL/XLSX/MP4 当前 SHA-256 已写入处理清单；其中 Stage 2 曾记录哈希的 **104** 个 PKL/XLSX 与本次一致，未发现 raw 内容变化。

## 尚未解决，不能据此进入 Pilot

1. 附件3无显式长度、ID、人工缺失掩码；边界零段及视觉原生零值的 `P/O` 仍为未知。`zero_context=3` 只能作候选诊断。
2. 附件3没有预计算 `text`：aligned 提供 `text_bert`，unaligned 提供 `raw_text`。本次只安全规范 dtype/保留原文本，没有替用户选择或训练文本编码器；未对附件3创造不存在的文本特征。
3. MP4 容器时长已验证，词、帧和特征到精确秒数的映射尚未核实；Q1/Q3 原始素材解释仍须继续证据恢复。

下一步应先审查本契约、报告和实际处理目录，再继续 Stage 2.5 的 observation/text/time 接口恢复。当前没有训练模型或进入 Stage 3。
