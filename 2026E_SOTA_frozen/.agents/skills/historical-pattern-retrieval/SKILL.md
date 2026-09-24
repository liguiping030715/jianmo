---
name: historical-pattern-retrieval
description: Retrieve structurally similar historical Huawei Cup solution patterns for a new problem and produce an adaptation plan without copying historical formulas, results, or wording.
---

# Historical Pattern Retrieval

## Scope separation

This skill learns **competition modeling structure and reasoning patterns** from historical excellent papers.
It does not replace Stage 1.1 current-literature research and must not be used as scientific evidence for current SOTA claims.

## Principle

Retrieve by **problem structure**, not by topic keyword alone.

## Inputs

- `workspace/analysis/task_fingerprint.md`
- `workspace/knowledge/reusable_patterns/`
- `workspace/knowledge/same_problem_synthesis/`
- `workspace/knowledge/pattern_cards/`

## Similarity dimensions

Evaluate similarity along:

- prediction / optimization / evaluation / simulation / reconstruction / scheduling,
- data modality,
- sample size and supervision,
- temporal structure,
- spatial structure,
- physical/mechanistic constraints,
- graph/resource constraints,
- multiple data sources,
- missing labels,
- source-target distribution shift,
- uncertainty,
- sequential dependency between questions,
- final action or decision.

## Required output

Create `workspace/analysis/historical_transfer_plan.md` with:

1. current-problem fingerprint,
2. 3-6 relevant historical patterns,
3. evidence for structural similarity,
4. transferable reasoning components,
5. non-transferable details,
6. misleading analogies rejected,
7. candidate new-solution ideas inspired by patterns,
8. open questions that only current data can answer.

## Prohibitions

Do not:

- choose a model solely because a historical excellent paper used it,
- copy historical formulas without re-derivation,
- copy numerical thresholds,
- copy paper wording,
- assume a historical validation protocol fits the current data,
- treat historical performance as a benchmark for the current problem.
