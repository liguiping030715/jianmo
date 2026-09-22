# P3 reusable_patterns: 迁移学习 / 跨域 / 域适应（带适用条件）

> P1.5 据 2025 E/B 深读重写。

---

## Pattern D-1：先确认分布偏移真实存在
- **Applicable When**：拟用迁移/域适应前。
- **Why It Works**：用 KDE 对比分布、MMD 量化距离、t-SNE 看双簇分离，确认偏移客观存在再处理，避免无谓对齐。
- **Evidence**：2025E 迁移前源/目标双簇分离、目标域特征数值差异显著。
- **Failure Modes**：偏移源于标注口径/预处理差异而非真实域差，对齐反而有害。
- **Do Not Use When**：两域分布本就一致时，域适应是多余的。

## Pattern D-2：对抗对齐 + 分布距离双机制
- **Applicable When**：存在显著域偏移，且需在对齐的同时保留类别判别力。
- **Why It Works**：DANN 的 minimax 隐式对齐 + MMD 显式约束分布距离，单一机制不稳时互补。
- **Evidence**：2025E DANN+MMD，MMD 0.89→0.15，分类器复用 ResNet-CBAM。
- **Failure Modes**：过度对齐把对分类有用的域特有信息抹掉；对抗训练不稳定。
- **Do Not Use When**：域差异极大、共享结构很少时，强行对齐得到伪对齐，应考虑重构特征或多源域。

## Pattern D-3：无标签目标域聚类 + 伪标签
- **Applicable When**：目标域完全无标签，但类别集合与源域对应。
- **Why It Works**：无监督聚成类，再用余弦/模板匹配对齐到源域已知类，迭代精化伪标签。
- **Evidence**：2025E 目标域聚类+余弦距离匹配，标定 16 个文件。
- **Failure Modes**：无真值时无法确认真实正确率，MMD/内聚性/匹配度只是代理指标；聚类数错误则全盘错。
- **Do Not Use When**：目标域出现源域没有的新类（开集），闭集匹配会强行错配。

## Pattern D-4：跨域防泄漏与可解释
- **Applicable When**：跨域特征处理与结果审计。
- **Why It Works**：标准化只用源域统计量、MIC 筛跨域共性特征；可解释三段（事前模块化/过程 t-SNE+MMD/事后 SHAP）。
- **Evidence**：2025E 源域 μ/σ 标准化、MIC 筛选、SHAP/TCAV 证据链。
- **Failure Modes**：用目标域统计量即信息泄漏；解释只证明"关注了特征"不证明分类对。
- **Do Not Use When**：无（防泄漏是默认；可解释不能替代对无标签结果的真值核验）。
