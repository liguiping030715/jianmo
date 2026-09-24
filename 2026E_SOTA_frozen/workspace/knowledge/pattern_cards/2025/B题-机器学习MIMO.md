# Card: 2025B MIMO-OFDM 链路速率（EESM物理抽象 + Isotonic单调校准 + HGBR/RF + OOF/GroupKFold）

> **Depth: L3_DEEP（P1.5 全文深读，含完整源码）**

## 0. 元数据
- 题目：MIMO-OFDM 上行链路速率（MCS）预测，2 发 4 收？实际 2×4，122 子载波。

## 1. 问题指纹
- 表面：由信道状态信息（CSI/SINR）预测传输速率 MCS。
- 真正困难：①122 维子载波 SINR 如何降维成可映射的单一质量指标；②SINR→MCS 必须单调（信道越好速率越高）；③同设备/同文件样本高度相关，随机划分会泄漏；④标签 MCS 是离散量化值且类别频次悬殊。

## 2. 数学抽象链
```
122维 SINR(线性) → γ_i=10log10(SINR_i)
EESM: γ_eff = -α·log( mean_i exp(-γ_i/β) )       (指数聚合降维, 物理PHY抽象)
     → IsotonicRegression: γ_eff → MCS (分段单调, 保序)
     → HGBR 轻量堆叠补非线性残差 (对 eff_db 单调约束 monotonic_cst=[1,0,...])
融合: y = (1-w)·iso_pred + w·hgbr_pred
     OOF 权重闭式解: w = mean((y-iso_pred)·d)/var(d), d=hgbr-iso, clip[0,1]
Q2: RF + 线性/dB 域 15 个统计增强特征; Isotonic in-fold; raw/mapped 双口径
```

## 3. 为什么每个模型被选
- **EESM**：3GPP 系统级仿真最常用 PHY 抽象，计算轻、可跨场景，与"按目标 BLER 选 MCS"流程一致；指数聚合对"最差子载波"敏感（一个坏子载波拉低整体）。
- **IsotonicRegression**：天然保证单调、不预设函数形式，能在分布偏移/噪声下把回归值拉回合理区间。
- **HGBR（带单调约束）**：EESM 可解释但残差有结构，用梯度提升补非线性，同时约束主特征与输出同单调，降过拟合。
- **RF（Q2）**：特征含大量统计量，RF 稳健、不需缩放。
- 物理模型打底 + ML 补残差，参数少、样本需求低。

## 4. Baseline 与消融逻辑
- 基线：纯 Isotonic（EESM→MCS）；改进：+HGBR 残差；融合权重由 OOF 估计（不用同集预测，防权重虚高）。
- Q2 同时报 raw（连续）与 mapped（就近贴标签集）两套指标，口径与提交一致。

## 5. 参数选择逻辑
- EESM 网格：α∈[1,6] step0.2；β∈[0.1,20] 对数+线性并集约 40 点；粗搜→局部随机细化（30 次/seed，a±0.3，β×[0.8,1.25]）。
- GroupKFold n_splits=5，分组=sta_mac/file_source。
- HGBR 三配置小网格（leaf31/63，lr0.06–0.1，iter250–400，L2 0–0.02）。
- RF：n_estimators500、min_samples_leaf2、max_features sqrt。
- winsor 限幅 [-200,200]；Isotonic out_of_bounds=clip。

## 6. 验证策略（防泄漏是核心）
- **GroupKFold 按设备 MAC/文件分组**，而非随机——同设备样本不跨训练/验证。
- **OOF 预测**估计融合权重，避免同集信息泄漏。
- 逆频率样本权重 1/freq(MCS)，缓解高频 MCS 支配。
- 数值稳健：log-sum-exp 的 stable 实现（减 zmax、clip[-745,709]）、非有限值兜底中位数。

## 7. 敏感性/鲁棒性
- winsorize 限幅 + clip 处理极端 SINR；Isotonic 对噪声单调稳健；多 seed 局部细化防单点。

## 8. 图表修辞角色
| 图表 | 回答 |
|---|---|
| 数据变量分析表 | 每列含义/维度？ |
| EESM 网格 leaderboard | 哪组 αβ 最优？ |
| 预测结果表（按 MAC） | 各设备预测速率？ |
| 指标双 Y 轴/对比表 | raw vs mapped 差异？ |

## 9. 章节承接技法
- Q1 的 EESM/Isotonic 物理链路 → Q2 扩展为含统计增强特征的 RF，Isotonic 作为 in-fold 校准复用；两问共享"分组 CV + 单调校准 + 双口径"框架。

## 10. 弱点 / 可疑论断（深读发现）
1. **单一 (α,β) 跨信道有系统偏差**：β 本与调制/码率相关，论文自承应分 MCS 标定或升级 MIESM（基于互信息）。
2. **Isotonic 分段常数外推弱**，winsor 限幅稳住训练却牺牲极端低/高 SNR 区精度；预测表大量重复值（516.2/412.9）说明输出粒度很粗。
3. 就近取整引入量化误差；MCS 残差随 eff 增大而异方差，平方损失对厚尾不鲁棒。
4. file_source one-hot 在新文件/新域失效，且可能"按文件记忆"。
5. 122 子载波仅做统计切片+EESM 聚合，未建模频域相关/选择性衰落形状，也未用 TxBF（仅取 TxBF=0）。
6. 参考文献含 DeepSeek-R1、GPT-5，AI 辅助痕迹明显（注意原创性）。

## 11. 可迁移条件 vs 不可迁移
- **可迁移**：EESM 式指数聚合降维 + log-sum-exp 数值技巧；Isotonic 单调校准；GroupKFold 按实体分组防泄漏；OOF 融合权重闭式估计；GBDT 单调约束；raw/mapped 双口径；"物理抽象打底 + ML 补残差"（**前提：物理聚合可信且残差有结构**）。
- **不可迁移**：α/β 数值、MCS 标签集、设备/文件 one-hot；EESM 在频率选择性极强/无单调保证场景需换 MIESM。
