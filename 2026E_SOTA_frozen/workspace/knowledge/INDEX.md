# Knowledge Base 总索引

> **分级口径（重要，防止虚假自信）**：45 篇 2024–2025 华为杯优秀论文**全部完成 L1 摘要级挖掘**；其中 **12 篇代表性论文已升级到 L3 全文深读**。**不得表述为"45 篇全部消化"。**

## 挖掘深度分级
| 级别 | 读取范围 | 掌握内容 |
|---|---|---|
| **L1_ABSTRACT** | 摘要、目录 | 整体路线、主要模型、各问关系、核心结果 |
| **L2_SECTION** | 问题分析/模型建立/结果分析/模型检验 | 为什么这样建模、baseline 设置、实验设计、如何论证有效 |
| **L3_DEEP** | 再读公式推导、参数、图表、敏感性、局限、附录/代码 | 真正可迁移的技术细节与失败模式 |

## 进度
- P1 Pattern Cards：45/45 L1 ✅；其中 **12/45 L3_DEEP** ✅
- P2 同题比较：12/12 ✅
- P3 跨题归纳：6/6（P1.5 已改为带适用条件的模式卡）✅
- P4 写作模式：3/3（P1.5 已删"至少"、图表按论证任务组织）✅

## 12 篇 L3_DEEP 清单（每题 1 篇代表）
**2024**
- A：`2024/A24103350007.md`（浙大，载荷估计/雨流-Miner/NSGA-II）
- B：`2024/B24103530099.md`（浙工商，DBSCAN+过采样 CNN/双层交叉预测）
- C：`2024/C24106130096.md`（西南交大，SE 方程 LSE/TLSE/消融矩阵）
- D：`2024/D24104250063.md`（石油大，MK+Sen/多源回归/HEV+AHP）
- E：`2024/E24102870008.md`（南航，YOLOv8/时空 SVR/Brlion 容量/决策防抖）
- F：`2024/F24103360083.md`（武大，轨道力学/Roemer/Shapiro/泊松仿真）

**2025**
- A：`2025/A题-2-核内调度.md`（贪心+禁忌/ABQPSO/DPEA 双种群）
- B：`2025/B题-机器学习MIMO.md`（EESM+保序映射/HGBR/OOF+GroupKFold）
- C：`2025/C题-3-围岩裂隙.md`（6 法投票/多层次聚类/傅里叶+小波/信息衰减布孔）
- D：`2025/D题-2-多源融合.md`（双模型蒸馏/变分同化三代价/各向异性/A*）
- E：`2025/E题-3-智能诊断.md`（手工+ViT 特征/ResNet-CBAM/DANN+MMD/可解释链）
- F：`2025/F题-量化分析.md`（几何代理/骨架图/适量度指标/开阔围合/双相似度）

> 其余 33 张卡片当前为 L1_ABSTRACT，按需再升级。

## 文件清单
- pattern_cards/2024/ — 24 张（6 张 L3，18 张 L1）
- pattern_cards/2025/ — 21 张（6 张 L3，15 张 L1）
- same_problem_synthesis/ — 12 份（2024 A-F + 2025 A-F）
- reusable_patterns/ — 6 份（prediction/optimization/evidence_gate/signal_image_space/domain_adaptation/evaluation）
- writing_patterns/ — 3 份（abstract/question_section/validation）

## 下一步：严格 Hold-out 盲测
- 选一个 hold-out 题（如 2024 E），测试时**禁止访问**该题的 pattern_cards 与 same_problem_synthesis（并尽量隔离由该题贡献的专有知识）。
- 只给原始题目 + 附件 + 其余历史库，独立完成全流程。
- 完成后解锁该题优秀论文，对比四件事：①有无漏题；②问题链是否合理；③核心数学抽象是否接近优秀方案；④验证是否充分。
