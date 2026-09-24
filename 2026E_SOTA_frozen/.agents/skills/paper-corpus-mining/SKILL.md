---
name: paper-corpus-mining
description: Systematically read all historical Huawei Cup excellent-paper Markdown files and extract transferable problem-solving, validation, visualization, and writing patterns. Use during precompetition learning, not as a substitute for solving the current problem.
---

# Paper Corpus Mining

## Goal

Turn the historical paper corpus into a structured case-based reasoning library.
The unit of learning is not the algorithm name. It is the **reasoning pattern**.

## Inputs

- `knowledge_sources/jianmo/Markdown/`
- `workspace/knowledge/corpus_manifest.csv`
- `templates/historical_pattern_card.md`
- `templates/same_problem_comparison.md`

## P0 — inventory

Verify that every Markdown paper is represented in the manifest.
Record malformed or partially converted files rather than silently excluding them.

## P1 — one card per paper

Read enough of the paper to understand:

- abstract,
- overall route,
- each question's analysis/model/result,
- validation or model evaluation,
- conclusion,
- important figures/tables when text explains their role.

Do not create a card from the abstract alone.

For each paper write a card using `templates/historical_pattern_card.md`.

A card must answer:

1. What was the problem's real structure?
2. How were sub-questions linked?
3. What intermediate mathematical representations were created?
4. Why did each model fit the data/mechanism?
5. Was there a baseline or controlled comparison?
6. What was actually validated?
7. What uncertainty or robustness issue was addressed or omitted?
8. What role did major figures/tables play?
9. How was the abstract organized?
10. How were question sections organized?
11. Which patterns are transferable to unrelated future problems?
12. Which details are specific to this historical problem and must not transfer?
13. What possible weaknesses should not be imitated?

## P2 — same-problem synthesis

Compare all excellent papers for the same historical problem.

Extract:

- shared backbone,
- alternative valid representations,
- alternative model families,
- conditions that explain different choices,
- recurring validation expectations,
- recurring writing structure,
- unique optional ideas.

Do not declare a universal winner.

## P3 — cross-problem synthesis

Cluster cases by structural archetype, not subject-matter keyword.

Examples:

- mechanism -> estimate -> optimize,
- baseline -> diagnose weakness -> improve,
- image/signal -> object representation -> decision,
- multi-source -> align -> fuse -> forecast -> optimize,
- domain shift -> adapt -> interpret,
- abstract concept -> measurable proxy -> index -> validate,
- DAG/resource constraints -> feasible scheduling -> multiobjective tradeoff.

## P4 — writing grammar

Infer recurring information architecture for:

- abstracts,
- problem analysis,
- model derivation,
- result presentation,
- validation,
- mini-summaries,
- model evaluation.

Never save copied sentences as templates.

## Quality gate

A paper card is incomplete if it only lists models.
A high-quality card must explain the chain:

`difficulty -> representation -> method -> evidence -> conclusion`.
