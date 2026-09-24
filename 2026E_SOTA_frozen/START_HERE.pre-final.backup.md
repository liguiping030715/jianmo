# HuaweiCup Codex v4 — 使用顺序

这个版本分成两个模式：**赛前学习** 和 **2026 正式比赛**。

## 一、现在：先把历年优秀论文变成方法库

仓库来源：

`https://github.com/liguiping030715/jianmo`

在项目根目录运行：

```powershell
python scripts/sync_jianmo.py
python scripts/build_corpus_manifest.py
```

然后对 Codex 输入：

> Read `AGENTS.md` and `.agents/skills/paper-corpus-mining/SKILL.md`.
> Use every Markdown paper under `knowledge_sources/jianmo/Markdown/`.
> Execute P0 and P1. Create one complete pattern card for every paper listed in
> `workspace/knowledge/corpus_manifest.csv`. Do not skip papers. Learn problem
> decomposition, question dependencies, intermediate mathematical representations,
> model-selection rationale, baseline/comparison design, validation, robustness,
> uncertainty, figure roles, abstract structure and section structure. Do not copy
> historical wording, formulas, numerical answers, or conclusions into a new problem.

P1 完成后运行：

```powershell
python scripts/check_pattern_coverage.py
```

必须看到所有历史论文都有 pattern card，才能进入 P2。

然后对 Codex 输入：

> Execute P2 same-problem synthesis for every year/question group, then P3
> cross-problem synthesis and P4 writing-grammar synthesis. Focus on what is
> structurally transferable to an unseen 2026 problem and explicitly record
> non-transferable problem-specific details.

## 二、2026 正式比赛开始前

当前 `workspace/` 中保留了你的历史练习工作。不要让旧题污染正式赛题。

确认要开始正式比赛后，运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/archive_practice_and_reset.ps1
```

它会备份当前练习 workspace，然后建立干净的比赛 workspace。

把 2026 赛题放到：

`workspace/problem/`

把原始附件放到：

`workspace/data/raw/`

## 三、2026 开题第一条 Codex 指令

> Read `AGENTS.md`. Treat the current files under `workspace/problem/` and
> `workspace/data/raw/` as a brand-new competition problem. Run Stage 1 only.
> Produce `problem_breakdown.md`, `variables_and_constraints.md`, `assumptions.md`
> and `task_fingerprint.md`. Do not choose a final model yet.

然后：

> Run Stage 1.2 historical pattern retrieval. Retrieve structurally similar
> patterns from `workspace/knowledge/`, not merely papers with similar keywords.
> Produce `historical_transfer_plan.md`, including useful analogies and rejected
> misleading analogies. Do not transplant historical formulas or results.

然后：

> Run Stage 1.5 solution architecture. Produce `solution_story.md`,
> `question_dependency_map.md`, and `intermediate_representations.md`. Make each
> sub-question's output feed the next where the problem structure supports it.
> Stop before full model implementation.

## 四、之后的固定流程

```text
读题
 -> 历史模式检索
 -> 解题架构
 -> 数据审计
 -> 必要时证据恢复
 -> baseline + 候选模型
 -> 小规模 Pilot
 -> Reviewer
 -> 正式实验
 -> 敏感性/鲁棒性/不确定性
 -> 图表
 -> 论文
 -> Final Audit
```

## 五、一个重要原则

历年优秀论文用于学习：

- 怎么拆题；
- 怎么把现实概念变成数学对象；
- 怎么让各问形成链条；
- 怎么设计 baseline 和改进；
- 怎么验证；
- 怎么组织摘要和正文。

不是用于：

- 看见同类关键词就套同一个算法；
- 复制公式、结果、文字；
- 把历史模型当成 2026 的默认答案。
