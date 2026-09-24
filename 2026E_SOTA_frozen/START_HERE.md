# HuaweiCup Codex Final — 2026 正式使用顺序

这个版本把三种能力分开：

1. **官方赛题/数据**：决定真正要解决什么；
2. **当前顶会顶刊文献**：了解最新科学机制；
3. **历年优秀数模论文 Pattern Library**：学习拆题、结构、验证与写作。

最终模型必须由当前赛题证据决定。

---

## 0. 第一次打开项目：同步最新版历史库

运行：

```powershell
python scripts/sync_jianmo.py
```

这个最终版会同时：

- clone/pull `jianmo`；
- 检查 `Markdown/` 历史论文；
- 把仓库中最新 `workspace/knowledge/` Pattern Library 合并到本地 `workspace/knowledge/`。

因此不会只停留在 ZIP 自带的 seed patterns。

可选检查：

```powershell
python scripts/check_pattern_coverage.py
```

---

## 1. 正式赛题开始：清理旧练习工作区

```powershell
powershell -ExecutionPolicy Bypass -File scripts/archive_practice_and_reset.ps1
```

旧 workspace 会被归档；`workspace/knowledge/` 保留。

把官方题面放到：

`workspace/problem/`

把官方原始数据/附件放到：

`workspace/data/raw/`

不要修改 raw。

---

## 2. 正式流程

```text
Stage 1     Problem Analysis
   ↓
Stage 1.1   Current SOTA Literature Scan
   ↓
Stage 1.2   Historical Pattern Retrieval
   ↓
Stage 1.3   Negative-Transfer & Evidence Audit
   ↓
Stage 1.5   Solution Architecture + Validation Blueprint
   ↓
[人工/Reviewer审架构]
   ↓
Stage 2     Data Audit
   ↓
Stage 3     Baseline + Candidate Models
   ↓
Stage 4     Pilot
   ↓
Stage 5     Reviewer
   ↓
Stage 6-10  Full experiments → validation → figures → paper → final audit
```

不要从 Stage 1 直接跳到“Transformer / XGBoost / GA”。

---

## 3. 2026 E 题

如果你现在做的是 2026 E：

直接打开根目录：

`PROMPT_2026E_START.md`

把里面的 Prompt 整段发给 Codex。

第一轮只做到 Stage 1–1.5，先不要训练最终模型。

---

## 4. Stage 1.1 与历史库的区别

```text
Stage 1.1 Current SOTA
→ 学：学术界现在如何解决这个科学问题

Stage 1.2 Historical Pattern Library
→ 学：优秀数模论文如何拆题、组织、验证和写作
```

二者都不能替代当前赛题证据。

---

## 5. 最重要的原则

**Current evidence decides the model.**

- 顶会方法不是因为“顶会”就采用；
- 历史优秀论文不是因为“优秀”就照搬；
- 更复杂不等于更好；
- 每个改进都要有 baseline、原因、pilot 和证伪条件；
- 文献检索要限时，不能把比赛做成 survey。
