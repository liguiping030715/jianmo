---
name: sota-literature-scan
description: Search, evaluate, and synthesize recent peer-reviewed literature for scientific subproblems identified in the current mathematical-modeling problem. Use after problem analysis and before historical retrieval and solution architecture.
---

# SOTA Literature Scan

## Purpose

Map the current scientific solution space without searching for competition-specific solutions.
The objective is **mechanism discovery and evidence**, not architecture copying.

## Required inputs

Read first:

- current official problem statement and attachment descriptions,
- `workspace/analysis/problem_breakdown.md`,
- `workspace/analysis/task_fingerprint.md`,
- current competition restrictions relevant to data/tools when available.

Use Stage 1 findings to formulate search queries. Do not begin from an algorithm name unless Stage 1 gives a reason.

## Search priority

Prefer recent peer-reviewed work, normally the latest 2–3 years when relevant.
Prefer primary sources:

1. official conference/journal proceedings,
2. CVF Open Access / ACL Anthology / IEEE / ACM / publisher pages,
3. accepted-author versions,
4. arXiv only when a peer-reviewed version is unavailable.

Choose venue families appropriate to the actual scientific problem. Typical high-quality sources may include CVPR/ICCV/ECCV, ACL/EMNLP/NAACL, ACM MM, AAAI/IJCAI, NeurIPS/ICML/ICLR, IEEE TPAMI/TMM/TASLP, Information Fusion, and strong domain-specific journals.

Venue prestige is not evidence of fit.

## Competition-integrity restriction

Never search for or use:

- solutions to the current competition problem,
- blogs/threads discussing the current problem,
- other teams' approaches,
- target-specific solution repositories,
- answer summaries,
- target-specific code produced as a competition solution after release.

Search the underlying scientific problem only.

## For every serious paper/method extract

- title,
- venue/year,
- stable source URL/DOI when available,
- exact scientific task,
- dataset/setup,
- input representation,
- core mechanism,
- objective/loss when relevant,
- assumptions,
- validation protocol,
- reported limitations,
- external-data requirement,
- compute/dependency requirements,
- reproducibility status,
- compatibility with current official data,
- transferable mechanism,
- non-transferable details,
- falsification condition for current use.

Distinguish **source-supported facts** from your own synthesis.

## Method classification

Classify each serious method as:

### DIRECTLY_USEFUL
Mechanism, assumptions, data interface, and competition feasibility closely match the current problem.

### USEFUL_COMPONENT_ONLY
One component or discipline transfers, but the full architecture does not.

### NOT_SUITABLE_FOR_THIS_COMPETITION
Task assumptions, data requirements, compute, dependencies, or validation mismatch the current problem.

## Feasibility gate

Explicitly evaluate:

- training/inference time,
- GPU/CPU/memory needs,
- preprocessing cost,
- dependency risk,
- need for external data,
- reproduction difficulty,
- parameter/model size when relevant,
- whether a simpler baseline could test the same hypothesis.

## Required outputs

Create:

- `workspace/references/sota_literature_scan.md`
- `workspace/references/sota_method_matrix.md`

Optionally maintain:

- `workspace/references/references.bib`

The matrix should include at least:

`paper | venue/year | problem | mechanism | assumptions | current-data compatibility | compute | external data | validation | advantage | risk | transferable component | non-transferable component | falsification | classification`

## End condition

Do not implement a final architecture during this skill.
Do not select a method solely because it is SOTA.

End with at most:

- one simple baseline direction,
- two serious candidate mechanism directions per major modeling stage,

and a clear list of open questions that only current-data audit/pilots can answer.
