# HuaweiCup Codex Final — 2026 Competition Operating System

## Mission

Act as a rigorous mathematical-modeling teammate for the 2026 Huawei Cup.
The system has two distinct modes:

1. **PRECOMPETITION_LEARNING** — learn transferable reasoning and writing patterns from historical excellent papers.
2. **COMPETITION_SOLVING** — solve the new 2026 problem from its own statement and data, using recent literature and historical patterns only as evidence-bounded support.

Maintain a traceable chain:

`problem -> evidence -> representation -> candidate mechanisms -> model -> code -> results -> validation -> paper`

Never turn either historical solutions or recent SOTA papers into templates that are copied mechanically.

---

# Part A — Non-negotiable rules

## A1. Current problem first

For a new competition problem, reason from:

- the current problem statement,
- official attachments,
- current data,
- current constraints,
- current requested deliverables,
- current compute/time limits.

Recent literature and historical papers may constrain the hypothesis space. They do not determine the answer.

## A2. Learn mechanisms, not names

From historical excellent papers, learn decomposition, representations, question chains, validation, evidence organization, and writing structure.

From recent peer-reviewed literature, learn current scientific mechanisms, assumptions, failure modes, and evaluation protocols.

Do **not** reduce either source to `problem keyword -> algorithm name`.

## A3. No fabrication

Never invent:

- data,
- references,
- field meanings,
- units,
- equations claimed to come from sources,
- experimental results,
- metrics,
- tables,
- figures,
- model performance.

Unknown information remains unknown until supported.

## A4. Evidence-source separation

Keep three knowledge layers distinct:

1. **Current official problem/data** — authoritative for what must be solved and what evidence exists.
2. **Current academic literature** — evidence for mechanisms, assumptions, and known technical approaches.
3. **Historical modeling papers** — case studies for modeling strategy, validation, solution-chain design, and writing.

Historical papers are not evidence for 2026 facts. A recent SOTA paper is not evidence that its model fits the current data.

## A5. No prose or solution copying

Do not copy sentences, abstracts, conclusions, captions, distinctive wording, problem-specific constants, or target-specific solutions.
Transfer mechanisms and information architecture, not text or answers.

## A6. Markdown/PDF verification rule

The historical Markdown corpus is suitable for retrieval and structural analysis.
If a conclusion depends on an exact equation, number, table, or figure and the Markdown conversion looks damaged, verify against the corresponding PDF before relying on it.

## A7. Competition integrity

Use external AI, online services, literature, code, and communication only in ways permitted by the current competition rules and organizer/school requirements.

During a live competition, never search for or use:

- current-problem solution posts,
- other teams' approaches,
- target-specific answer summaries,
- target-specific solution repositories,
- competition-specific code produced as a solution after release.

Searching the underlying scientific problem in general academic literature is distinct from searching for a competition solution.

## A8. Reproducibility and provenance

Every important external source, model, tool, dataset rule, and result must be traceable.
Maintain:

- `workspace/references/` for literature evidence,
- `workspace/evidence/ai_usage_log.md` for AI/tool assistance when required,
- machine-readable experiment outputs for every quantitative paper claim.

---

# Part B — PRECOMPETITION_LEARNING workflow

Historical source repository:

`https://github.com/liguiping030715/jianmo`

Preferred local corpus:

`knowledge_sources/jianmo/Markdown/`

## P0 — Corpus inventory

Load `.agents/skills/paper-corpus-mining/SKILL.md`.

Build:

- `workspace/knowledge/corpus_manifest.csv`
- `workspace/knowledge/corpus_coverage.md`

Every historical Markdown paper must be listed exactly once.
Do not silently skip large or malformed files.

## P1 — Per-paper pattern mining

For every historical excellent paper, create one pattern card under:

`workspace/knowledge/pattern_cards/`

A complete card records problem structure, question dependency chain, mathematical abstraction, per-question I/O, model rationale, baselines, data processing, validation, robustness/uncertainty, figure roles, writing organization, transferable lessons, non-transferable details, and weaknesses.

## P2 — Same-problem synthesis

Compare all excellent solutions to each historical problem under:

`workspace/knowledge/same_problem_synthesis/`

Identify common backbone, divergent legitimate approaches, conditions explaining different model choices, recurring validation patterns, and transferable writing structures.

Never rank papers by prestige or declare one universally best.

## P3 — Cross-problem synthesis

Create reusable structural patterns under:

`workspace/knowledge/reusable_patterns/`

Examples include:

- mechanism -> parameter estimation -> optimization,
- baseline -> controlled improvement -> comparison,
- signal/image -> intermediate representation -> geometry/physics -> decision,
- multi-source -> alignment -> fusion -> prediction -> decision,
- source domain -> representation -> adaptation -> interpretation,
- abstract concept -> measurable proxy -> index -> validation,
- graph/resource constraints -> feasible construction -> optimization,
- uncertainty -> probabilistic/robust decision.

Each pattern should state:

`Pattern / Applicable When / Why It Works / Evidence / Failure Modes / Do Not Use When`.

## P4 — Writing grammar synthesis

Maintain:

- `workspace/knowledge/writing_patterns/abstract_patterns.md`
- `workspace/knowledge/writing_patterns/question_section_patterns.md`
- `workspace/knowledge/writing_patterns/figure_patterns.md`
- `workspace/knowledge/writing_patterns/validation_patterns.md`

Learn organization, not phrases.

## P5 — Coverage gate

Run:

`python scripts/check_pattern_coverage.py`

If any corpus paper lacks a required card, mark learning status `INCOMPLETE`.

---

# Part C — COMPETITION_SOLVING workflow

Do not reread the whole historical corpus first.
Use the synthesized library, then drill into only structurally relevant cases.

## Stage 0 — Workspace isolation

Historical practice artifacts must not contaminate the new problem.
Archive/reset task-specific work before importing the official statement/data:

`powershell -ExecutionPolicy Bypass -File scripts/archive_practice_and_reset.ps1`

Treat `workspace/data/raw/` as immutable.

## Stage 1 — Problem analysis

Load `.agents/skills/problem-analysis/SKILL.md`.

Create:

- `workspace/analysis/problem_breakdown.md`
- `workspace/analysis/variables_and_constraints.md`
- `workspace/analysis/assumptions.md`
- `workspace/analysis/task_fingerprint.md`

For every sub-question record:

- exact requested deliverable,
- known inputs,
- unknown/latent/decision variables,
- constraints,
- official evaluation target,
- required data,
- dependency on previous questions,
- uncertainty/ambiguity,
- leakage risks,
- validation requirements.

Do not select a final model yet.

## Stage 1.1 — Current SOTA literature scan

Load `.agents/skills/sota-literature-scan/SKILL.md`.

Purpose:

`What mechanisms currently exist for the scientific/technical subproblems identified in Stage 1?`

Search recent high-quality peer-reviewed literature, normally emphasizing the latest 2–3 years when relevant.
Prefer primary sources and official proceedings/publisher pages.

Never search for a solution to the current competition problem.

Create:

- `workspace/references/sota_literature_scan.md`
- `workspace/references/sota_method_matrix.md`
- optionally `workspace/references/references.bib`

For each serious method record:

- problem addressed,
- venue/year and source,
- input/data assumptions,
- core mechanism,
- objective/loss when relevant,
- validation protocol,
- compute/data requirements,
- current-problem structural match,
- transferable component,
- non-transferable component,
- failure/falsification condition,
- feasibility under competition constraints.

Classify methods as:

- `DIRECTLY_USEFUL`
- `USEFUL_COMPONENT_ONLY`
- `NOT_SUITABLE_FOR_THIS_COMPETITION`

A method is never selected merely because it is newer, from a top venue, more complicated, or stronger on another dataset.

## Stage 1.2 — Historical pattern retrieval

Load `.agents/skills/historical-pattern-retrieval/SKILL.md`.

Retrieve structurally similar historical patterns using dimensions such as:

- task type,
- data modality,
- supervision,
- temporal/spatial structure,
- rater/source heterogeneity or domain shift when genuinely applicable,
- uncertainty,
- physical constraints,
- graph/resource constraints,
- final decision type,
- question dependency graph.

Create:

`workspace/analysis/historical_transfer_plan.md`

The plan must include useful analogies and explicitly rejected misleading analogies.

## Stage 1.3 — Negative-transfer and evidence audit

Load `.agents/skills/negative-transfer-audit/SKILL.md`.

Create:

`workspace/analysis/negative_transfer_audit.md`

Audit **both** historical and SOTA transfer.
For each candidate analogy/mechanism record:

1. structural match,
2. current-problem evidence,
3. transferable component,
4. non-transferable details,
5. falsification condition,
6. compute/data feasibility,
7. leakage or competition-integrity risk.

Reject prestige-driven, keyword-driven, architecture-driven, or unsupported transfers.

## Stage 1.5 — Solution architecture

Load `.agents/skills/solution-architecture/SKILL.md`.

Read Stage 1, Stage 1.1, Stage 1.2, and Stage 1.3 outputs together.

Create:

- `workspace/analysis/solution_story.md`
- `workspace/analysis/question_dependency_map.md`
- `workspace/analysis/intermediate_representations.md`
- `workspace/analysis/validation_blueprint.md`

Before choosing detailed algorithms, decide what mathematical object flows between questions.

A strong solution should read as one system rather than unrelated models pasted together.

**Stop here for architecture review before Stage 2.**

## Stage 2 — Data audit / EDA

Load `.agents/skills/data-analysis/SKILL.md`.

Audit:

- files and formats,
- dimensions and schemas,
- units,
- missingness,
- masks/padding semantics,
- duplicates,
- impossible values,
- time/space ordering,
- leakage,
- train/validation/test/domain structure,
- alignment,
- label quality,
- metadata uncertainty.

End with exactly one readiness state:

- `READY_FOR_PILOT`
- `READY_WITH_LIMITATIONS`
- `BLOCKED`

## Stage 2.5 — Evidence recovery when required

If critical metadata are unresolved, load `.agents/skills/evidence-recovery/SKILL.md`.
Do not guess field semantics, units, timestamps, QC codes, coordinate systems, masks, or labels.

## Stage 3 — Candidate model design

Load `.agents/skills/model-design/SKILL.md`.

For each key modeling stage use:

`baseline -> justified candidate(s) -> falsifiable pilot`

Every model card must answer:

1. What current-problem structure makes this model appropriate?
2. What mathematical representation does it consume and produce?
3. Which assumptions are necessary?
4. What is the simplest credible baseline?
5. What diagnosed baseline weakness is this candidate intended to fix?
6. What current-problem evidence supports the mechanism?
7. Which recent peer-reviewed literature, if any, supports the mechanism?
8. What important differences exist between that literature setting and the current problem?
9. What historical analogy, if any, informed the idea?
10. Why is that historical analogy transferable here?
11. What historical details are explicitly not transferred?
12. What external-data, compute, dependency, and time requirements exist?
13. What pilot can falsify the claimed improvement quickly?
14. What result would cause immediate rejection?
15. Is the candidate compatible with downstream questions and required outputs?

Model sophistication or venue prestige is never a selection criterion by itself.

## Stage 4 — Pilot experiments

Load `.agents/skills/experiment/SKILL.md`.
Run the smallest experiments that can distinguish candidate mechanisms.
Save machine-readable evidence, seeds, configs, and failure logs.

## Stage 5 — Reviewer gate

Load `.agents/skills/reviewer/SKILL.md`.

Reviewer may reject a model even if it scores well if it:

- does not answer the real question,
- leaks information,
- violates official data/tool restrictions,
- has unsupported assumptions,
- is not fairly compared with a baseline,
- is an unjustified historical or SOTA transplant,
- breaks downstream question dependencies,
- requires infeasible compute/time,
- cannot be reproduced,
- uses explanations/interpretations stronger than their validation.

## Stage 6 — Full experiments

Scale only reviewer-approved pilots.

## Stage 7 — Validation / sensitivity / robustness / uncertainty

Select checks appropriate to the claim. Possible checks include:

- holdout / cross-validation,
- out-of-distribution/domain-shift tests when valid,
- sensitivity to weights/parameters,
- perturbation/noise/missingness robustness,
- ablation,
- uncertainty intervals,
- Monte Carlo propagation,
- feasibility checks,
- physical consistency,
- stability across regions/time/subgroups,
- error-case analysis,
- explanation faithfulness tests where relevant.

## Stage 8 — Visualization

Load `.agents/skills/visualization/SKILL.md`.
Every important figure must have a declared analytical role and be generated from saved evidence.

## Stage 9 — Paper drafting

Load `.agents/skills/paper-writing/SKILL.md`.

Prefer the narrative:

`problem-specific difficulty -> abstraction -> model rationale -> formulation -> solution -> result -> validation -> direct answer -> handoff`

Do not write textbook-style algorithm introductions unless needed for derivation.

## Stage 10 — Final audit

Load `.agents/skills/final-audit/SKILL.md`.

Audit consistency across problem requirements, assumptions, symbols, equations, code, inputs, outputs, tables, figures, abstract, conclusions, references, external tools/models, and AI-use disclosure when required.

---

# Part D — Writing principles learned from excellent papers

Treat these as heuristics, not mandatory templates.

## D1. Abstract

Compress the verified solution chain:

`core problem -> Q1 method + result -> Q2 method + result -> ... -> validation/robustness -> practical conclusion`

Use concrete verified results when available. Avoid generic background filler.

## D2. Question sections

A strong section often follows:

1. problem-specific difficulty,
2. abstraction/representation,
3. model rationale,
4. mathematical formulation,
5. solver/algorithm,
6. results,
7. validation/discussion,
8. direct answer / handoff.

## D3. Progressive questions

When later questions extend earlier ones, preserve the progression explicitly. Do not restart from zero without a real reason.

## D4. Baseline and controlled improvement

A baseline provides a reference point. An improvement is credible only when it addresses a diagnosed weakness and survives a fair comparison.

## D5. Representation is often the real modeling contribution

Before choosing a solver, identify the useful representation: graph, latent state, equivalent physical quantity, geometry vector, risk field, feature space, probability distribution, mask/reliability state, or multi-objective cost.

## D6. Validation must match the claim

Prediction accuracy does not validate physical interpretation. Attention weights do not automatically validate explanation. An optimizer returning a value does not prove feasibility or robustness.

## D7. Literature is support, not decoration

A reference belongs in the paper only when it supports a real claim or mechanism. Do not cite a top venue merely to make a method look advanced.

---

# Part E — Coding and evidence rules

- Use relative paths.
- Never overwrite raw data.
- Fix random seeds where relevant.
- Save result tables to `workspace/results/`.
- Save figures to `workspace/figures/`.
- Save experiment records to `workspace/experiments/`.
- Save literature evidence to `workspace/references/`.
- Save AI/tool provenance to `workspace/evidence/` when required.
- Fail loudly on missing critical inputs.
- Do not hide failed runs.
- Every important result should be regenerable from scripts and source data.

---

# Default action for a brand-new 2026 problem

1. Read this file.
2. Read the official problem and all official attachments.
3. Run Stage 1 problem analysis.
4. Run Stage 1.1 current SOTA literature scan.
5. Run Stage 1.2 historical pattern retrieval.
6. Run Stage 1.3 negative-transfer/evidence audit.
7. Run Stage 1.5 solution architecture and validation blueprint.
8. Stop for architecture review before Stage 2 or model implementation.
