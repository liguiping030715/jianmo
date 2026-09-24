# Q3 Stage 2 — 解释结果分析与论文图表

**状态：Q3_STAGE2_ANALYSIS_COMPLETE**

范围：对冻结的 Q3 Stage 1 证据做仅推理分析。无模型更新，未访问 A2 test / 附件3 / 附件4。
冻结预测器：M0 seed2718（checkpoint SHA-256 `6b1d31d2af92a5edb84ea133350db98eb5a7a5fcdbece2debd4887b978e26c56`）。
样本：A2 unaligned valid 728 行。

---

## 1. 模态贡献分析

### 1.1 Donor 替换响应

将每个样本的某一模态张量替换为另一个随机样本（无自替换）的对应张量，测量预测变化。这是模型干预响应，不是因果效应。

| 模态 | 目标 | n | 均值 | 中位数 | P05 | P95 |
|---|---|---:|---:|---:|---:|---:|
| Text | cls | 728 | **0.9226** | 0.8367 | -1.0493 | 3.0111 |
| Text | reg | 728 | **0.7757** | 0.6474 | 0.0559 | 1.9372 |
| Audio | cls | 728 | 0.0301 | 0.0227 | -0.2819 | 0.3907 |
| Audio | reg | 728 | 0.0926 | 0.0578 | 0.0055 | 0.2860 |
| Vision | cls | 728 | 0.0517 | 0.0264 | -0.4031 | 0.5464 |
| Vision | reg | 728 | 0.1533 | 0.0873 | 0.0051 | 0.5941 |
| Audio+Vision joint | cls | 728 | 0.0815 | 0.0744 | -0.5012 | 0.6917 |
| Audio+Vision joint | reg | 728 | 0.1977 | 0.1357 | 0.0095 | 0.6254 |

**核心发现**：文本模态的贡献远大于音频和视觉。文本单独替换引起的分类 logit 变化均值（0.9226）是音频（0.0301）的 30.7 倍、视觉（0.0517）的 17.8 倍。即使将音频和视觉同时替换（0.0815），其效应仍不到文本单独的 1/11。

图：[q3_modality_contribution_boxplot.png](../../figures/q3_modality_contribution_boxplot.png)

### 1.2 主导模态分布

| 目标 | Text | Audio | Vision | 总计 |
|---|---:|---:|---:|---:|
| Classification | **551 (75.7%)** | 89 (12.2%) | 88 (12.1%) | 728 |
| Regression | **615 (84.5%)** | 37 (5.1%) | 76 (10.4%) | 728 |

在约 3/4 的样本中，文本是对分类预测影响最大的模态；在强度回归中这一比例升至 84.5%。音频和视觉仅在少数样本中成为主导模态。

图：[q3_dominant_modality.png](../../figures/q3_dominant_modality.png)

### 1.3 与 Q2 证据的呼应

Q2 factorial evidence 显示，文本缺失导致最明显的性能退化（M0 macro-F1 降 0.0349，MAE 升 0.0267），而音频/视觉缺失的平均效应接近零甚至微升。Q3 的模态贡献分析从模型内部响应角度独立验证了同一结论：**该冻结预测器主要依赖文本模态，音频和视觉提供辅助信息。**

---

## 2. 忠实度验证

对每个模态和目标，按单区间响应排序后选取 top / random / bottom 三个等数量互不重叠区间组，做联合删除对照。

| 模态 | 目标 | 合格样本 | Top 均值 | Random 均值 | Bottom 均值 | 通过 |
|---|---|---:|---:|---:|---:|---|
| Text | cls | 644 | 0.8089 | 0.4203 | 0.0103 | ✅ |
| Text | reg | 641 | 0.4620 | 0.3006 | 0.1718 | ✅ |
| Audio | cls | 728 | 0.0456 | 0.0014 | -0.0440 | ✅ |
| Audio | reg | 728 | 0.0146 | 0.0086 | 0.0019 | ✅ |
| Vision | cls | 726 | 0.0323 | 0.0042 | -0.0134 | ✅ |
| Vision | reg | 726 | 0.0268 | 0.0100 | 0.0014 | ✅ |

所有 6 个模态×目标组合均满足 top > random > bottom，且 paired 95% CI 不含零。这验证了解释算子的内部一致性：被单区间测试标记为"重要"的区间，在联合删除时确实引起更大的预测变化。

注意：rank-then-joint-test 仍是样本内排序，可能因构造而偏向 top；应理解为算子一致性检验，而非外部有效性。

图：[q3_faithfulness_top_random_bottom.png](../../figures/q3_faithfulness_top_random_bottom.png)

---

## 3. 稳定性分析

在预测保持不变（类别不变且强度变化 ≤0.1）的子样本上，用最低响应区间作为小扰动，重新评分剩余同尺度候选区间，测量 top-1 一致率和 top-3 Jaccard。

| 模态 | 目标 | 预测保持样本 | Top-1 一致率 | Top-3 Jaccard |
|---|---|---:|---:|---:|
| Text | cls | 56 | 0.643 | 0.627 |
| Text | reg | 115 | 0.565 | 0.549 |
| Audio | cls | 128 | **0.961** | 0.965 |
| Audio | reg | 127 | **0.992** | 0.976 |
| Vision | cls | 128 | **0.977** | 0.992 |
| Vision | reg | 128 | **0.992** | 1.000 |

**文本解释的稳定性明显低于音频和视觉。** 这可能因为文本扰动（`[UNK]` 替换）经过冻结 BERT 重新编码后，上下文表示发生较大变化，导致区间重要性排序重排。音频/视觉的局部区间删除不经过预训练编码器重编码，因此排序更稳定。

这一差异不影响解释方法的有效性，但意味着：**在论文中呈现文本局部解释时，应同时报告稳定性区间，避免将单次排序过度解读为确定的"最重要词"。**

图：[q3_stability.png](../../figures/q3_stability.png)

---

## 4. 局部证据分析

### 4.1 文本高影响短语

对 3-token 候选区间，按出现次数 ≥3 的短语聚合，按平均 |Δ分类 logit| 排序。Top 10：

| 短语 | 平均 |Δcls| | 出现次数 |
|---|---:|---:|
| wasn't | 1.1267 | 3 |
| don't | 0.7697 | 4 |
| doesn't | 0.6410 | 5 |
| a wide range | 0.6362 | 3 |
| 's kind of | 0.6179 | 4 |
| one of the | 0.5326 | 8 |
| it was a | 0.4900 | 3 |
| i didn' | 0.4873 | 3 |
| didn't | 0.4500 | 6 |
| umm ) it | 0.4292 | 7 |

**否定结构（wasn't / don't / doesn't / didn't）是对分类预测影响最大的文本片段类型。** 这符合情感分析的语义直觉：否定词翻转或减弱情感极性，因此删除后预测变化最大。口语化填充词（umm, kind of）也有较高影响，可能因为它们与特定情感表达风格相关。

注意：这些是 tokenizer 重构的短语，不是经过验证的口语时间戳。

图：[q3_top_text_phrases.png](../../figures/q3_top_text_phrases.png)

### 4.2 音频/视觉局部区间

| 模态 | 区间比例 | 平均 |Δcls| | 平均 |Δreg| | 候选数 |
|---|---:|---:|---:|---:|
| Audio | 1% | 0.0026 | 0.0015 | 8736 |
| Audio | 5% | 0.0082 | 0.0047 | 8736 |
| Audio | 10% | 0.0135 | 0.0078 | 8735 |
| Vision | 1% | 0.0013 | 0.0010 | 8666 |
| Vision | 5% | 0.0045 | 0.0036 | 8666 |
| Vision | 10% | 0.0081 | 0.0066 | 8663 |

音频/视觉的局部区间效应随区间长度单调增加，但绝对量级远小于文本。音频效应略大于视觉，与模态贡献分析一致。

---

## 5. 代表性案例

从每类主导模态中选取该模态 Δcls 最大的样本：

| 案例类型 | 样本 ID | 预测极性 | 主导模态 (cls) |
|---|---|---|---|
| Text-dominant | 50103$_$3 | Negative | text |
| Audio-dominant | 99501$_$10 | Negative | audio |
| Vision-dominant | HGicRLwgtkM$_$2 | Positive | vision |

完整数据见 [q3_representative_cases.csv](q3_representative_cases.csv)。

---

## 6. 局限性

1. **Donor 响应不是因果效应**：替换为另一个随机样本的张量，测量的是模型对样本特定内容的依赖程度，受 donor 选择影响。
2. **文本短语不是验证的口语时间戳**：tokenizer 重构的短语可能与实际发音区间不对应。
3. **音频/视觉仅用 native index**：无法转换为物理秒数，区间位置是相对比例。
4. **稳定性扰动是声明的小干预**：不是全面的鲁棒性测试；文本稳定性低部分源于 BERT 重编码的敏感性。
5. **无注意力权重归因**：不使用 attention 权重作为解释分数。
6. **无认证的隐藏缺失掩码**：P/O 状态来自清洗管线，不构成认证缺失。

---

## 7. 结论

Q3 Stage 2 对冻结 M0 seed2718 预测器的解释分析表明：

1. **文本是绝对主导模态**（75.7% 分类、84.5% 回归样本），其贡献远大于音频和视觉之和。这与 Q2 缺失鲁棒性证据独立呼应。
2. **解释算子通过忠实度检验**：所有 6 个模态×目标组合满足 top > random > bottom，算子内部一致。
3. **文本解释稳定性低于音频/视觉**：论文中呈现文本局部解释时应报告稳定性，避免过度解读单次排序。
4. **否定结构是最高影响文本片段**：wasn't / don't / doesn't / didn't 等否定词对情感极性判断影响最大。

这些结果可直接用于论文第三问的"模型可解释性分析"章节。

---

## 产物清单

**图表**（`workspace/figures/`）：
- `q3_modality_contribution_boxplot.png` — 模态 donor 响应箱线图
- `q3_dominant_modality.png` — 主导模态分布
- `q3_faithfulness_top_random_bottom.png` — 忠实度 top/random/bottom 对照
- `q3_stability.png` — 解释稳定性
- `q3_top_text_phrases.png` — 文本高影响短语 Top 20

**数据**（`workspace/results/q3_stage2/`）：
- `q3_modality_contribution_summary.csv`
- `q3_dominant_modality_summary.csv`
- `q3_faithfulness_summary.csv`
- `q3_stability_summary.csv`
- `q3_top_text_phrases.csv`
- `q3_local_av_summary.csv`
- `q3_representative_cases.csv`
- `q3_stage2_status.json`
