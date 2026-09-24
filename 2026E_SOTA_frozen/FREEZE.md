# 2026E SOTA — Frozen v1.0

**冻结日期**: 2026-09-24
**状态**: FROZEN — 此版本为实验基线，后续新版本在此基础上迭代

---

## 模型与性能

**冻结预测器**: M0 (NativeMulT) seed2718
- Checkpoint SHA-256: `6b1d31d2af92a5edb84ea133350db98eb5a7a5fcdbece2debd4887b978e26c56`
- 架构: text-query cross-attention to audio/vision, DIM=48, heads=2, dropout=0.1
- 训练: AdamW lr=0.001, lambda_reg=2.0526, batch=32, 3 seeds (1729/2718/31415)

### A2 官方测试集 (n=727, 一次性测试)

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.6726 |
| macro-F1 | 0.5987 |
| weighted-F1 | 0.6535 |
| MAE | 0.6274 |
| Pearson | 0.6729 |

### 各类别 (test)

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.648 | 0.720 | 0.682 | 207 |
| Neutral | 0.472 | 0.266 | 0.340 | 158 |
| Positive | 0.730 | 0.823 | 0.774 | 362 |

### A2 验证集三种子均值 (n=728)

| 模型 | Clean macro-F1 | Clean MAE | Gap F1 (S1-S6) | Gap MAE |
|---|---:|---:|---:|---:|
| M0 | 0.5812±0.0256 | 0.6060±0.0250 | 0.5588±0.0333 | 0.6223±0.0275 |
| M1 | 0.5717±0.0282 | 0.6012±0.0271 | 0.5570±0.0343 | 0.6187±0.0306 |

---

## 三问完成状态

### Q1 — 特征提取与时间对齐 ✅
- 100/100 样本成功, 78 VERIFIED / 22 UNRESOLVED
- 文本: bert-base-uncased + wav2vec2 forced alignment
- 音频: openSMILE eGeMAPSv02 LLD (25-dim)
- 视觉: MediaPipe FaceMesh (1404-dim)
- 统一 K=50 bin 对齐
- 输出: `workspace/q1_final/`

### Q2 — 鲁棒预测模型 ✅
- M0 冻结, 37 个实验 checkpoint (R0/R0a/R1/R2/C2/RU/M0/M1/D0/D1/E0/E1)
- 36 条件 factorial evidence (715 common cohort, 216 指标)
- 附件3 (30样本) 全量预测完成
- 缺失增强: S1-S6 六场景
- 输出: `workspace/results/q2_evidence/`, `workspace/results/q2_final/`

### Q3 — 可解释性 ✅
- 模态贡献: donor 替换 (5种), 文本主导 75.7% cls / 84.5% reg
- 局部证据: 滑窗遮挡 (文本3-token / 音视频5%)
- 忠实度: 6/6 模态×目标组合通过 (top > random > bottom)
- 稳定性: 文本 top1=0.56-0.64, 音视频 0.96-1.0
- 输出: `workspace/results/q3_stage1/`, `workspace/results/q3_stage2/`

---

## 已知缺口 (新版本待解决)

1. **Reliability Gate** — M0 无门控融合, 全缺模态无显式置零
2. **类别权重 CE** — Neutral 类 recall 仅 26.6%, 需 class_weight
3. **一致性损失** — 无 clean-gap 一致性正则
4. **单模态基线** — 缺文本-only/音频-only/视觉-only 基线
5. **Q3 精确 Shapley** — 当前用 donor 替换(5种), 文档要求8子集精确Shapley
6. **缺失增强 TAV** — 缺三模态同时缺失场景
7. **错误归因分析** — 无 confusion matrix 深度分析
8. **组件级消融表** — 缺标准消融链

---

## 排除的大文件 (未上传)

以下文件因体积过大未包含在 Git 中, 需从原始数据重新生成:

| 路径 | 大小 | 说明 |
|---|---:|---|
| `workspace/data/raw/` | ~3.7 GB | 附件2原始 pkl (unaligned_50.pkl 2.7GB, aligned_50.pkl 948MB) |
| `workspace/data/processed/` | ~2.5 GB | 清洗后 npy (text/audio/vision) |
| `workspace/experiments/pilot_cache/` | ~4.2 GB | 训练缓存 npy |
| `workspace/references/` | ~780 MB | BERT + wav2vec2 预训练权重 (可从 HuggingFace 下载) |
| `knowledge_sources/` | ~433 MB | 历史优秀论文 PDF (已在 jianmo 仓库根目录) |

### 复现步骤

1. 下载附件2原始数据到 `workspace/data/raw/E题数据/`
2. 运行清洗脚本生成 `workspace/data/processed/cleaning_2026e_v2/`
3. 下载 bert-base-uncased 和 wav2vec2-base-960h 到 `workspace/references/`
4. 运行 `pilot_ru.py` 训练 M0
5. 运行 Q1/Q2/Q3 实验脚本

---

## 目录结构

```
2026E_SOTA_frozen/
├── AGENTS.md                    # 项目操作系统
├── q2_bridge.py                 # M0 模型定义 (NativeMulT)
├── pilot_ru.py                  # 训练脚本 (数据加载+S1-S6+训练)
├── q2_unaligned_text_adapter.py # 文本适配器
├── q3_diagnostic.py             # Q3 诊断
├── PROMPT_2026E_START.md        # 启动 prompt
├── START_HERE.md
├── CHANGELOG_FINAL.md
├── .agents/                     # 专项 skills
├── scripts/                     # 工具脚本
├── templates/                   # 模板
└── workspace/
    ├── analysis/                # 分析文档
    ├── data_audit/              # 数据审计
    ├── evidence/                # 证据
    ├── evidence_recovery/       # 证据恢复
    ├── experiments/             # 实验代码 + checkpoint (37个)
    ├── figures/                 # 论文图表
    ├── knowledge/               # 知识库
    ├── models/                  # 模型卡片
    ├── problem/                 # 题目
    ├── q1_final/                # Q1 特征 + 报告
    └── results/                 # Q2/Q3 结果 (CSV/JSON/MD)
```
