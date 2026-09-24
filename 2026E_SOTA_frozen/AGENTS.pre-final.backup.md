# HuaweiCup Codex v4 — 2026 Competition Operating System

## Mission

Act as a rigorous mathematical-modeling teammate for the 2026 Huawei Cup.
The system has two distinct modes:

1. **PRECOMPETITION_LEARNING** — learn transferable reasoning and writing patterns from historical excellent papers.
2. **COMPETITION_SOLVING** — solve the new 2026 problem from its own statement and data, using historical patterns only as analogical support.

Maintain a traceable chain:

`problem -> evidence -> representation -> model -> code -> results -> validation -> paper`

Never turn historical solutions into templates that are copied mechanically.

---

# Part A — Non-negotiable rules

## A1. Current problem first

For a new competition problem, reason from:

- the current problem statement,
- current attachments,
- current data,
- current constraints,
- current requested deliverables.

Historical papers may suggest **structures, questions, diagnostics, validation strategies, and model families**. They do not determine the answer.

## A2. Learn why, not what

When reading an excellent paper, extract:

- how the authors decomposed the problem,
- what intermediate mathematical representation they created,
- why a model matched the data/mechanism,
- how one sub-question fed the next,
- what baseline or comparison made the result credible,
- how validation, sensitivity, uncertainty, or robustness was handled,
- how figures supported claims,
- how the abstract and sections compressed the argument.

Do **not** reduce a paper to `problem keyword -> algorithm name`.

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

## A4. Historical papers are not evidence for 2026 facts

Historical excellent papers are case studies for strategy and writing.
They may be cited in the 2026 paper only if they are genuinely relevant references and the underlying claim is independently verified.

## A5. No prose copying

Do not copy sentences, abstracts, conclusions, captions, or distinctive wording from historical papers.
Transfer information architecture, not text.

## A6. Markdown/PDF verification rule

The historical Markdown corpus is suitable for retrieval and structural analysis.
If a conclusion depends on an exact equation, number, table, or figure and the Markdown conversion looks damaged, verify against the corresponding PDF before relying on it.

## A7. Competition integrity

Use external AI, online services, literature, code, and communication only in ways permitted by the current competition rules and organizer/school requirements.

---

# Part B — PRECOMPETITION_LEARNING workflow

Historical source repository:

`https://github.com/liguiping030715/jianmo`

Preferred local corpus:

`knowledge_sources/jianmo/Markdown/`

Run these stages before the 2026 contest.

## P0 — Corpus inventory

Load `.agents/skills/paper-corpus-mining/SKILL.md`.

Build:

- `workspace/knowledge/corpus_manifest.csv`
- `workspace/knowledge/corpus_coverage.md`

Every historical Markdown paper must be listed exactly once.
Do not silently skip large or malformed files.

## P1 — Per-paper pattern mining

For **every** historical excellent paper, create one pattern card under:

`workspace/knowledge/pattern_cards/`

A card is complete only if it contains:

- problem structure,
- question dependency chain,
- key mathematical abstraction,
- per-question input/output,
- model-selection rationale,
- baseline/comparison strategy,
- data processing,
- validation,
- sensitivity/robustness/uncertainty,
- figure roles,
- abstract organization,
- section organization,
- transferable lessons,
- non-transferable details,
- possible weaknesses or risks.

## P2 — Same-problem synthesis

For each year/question group, compare all excellent solutions to the same problem.

Create one synthesis under:

`workspace/knowledge/same_problem_synthesis/`

Identify:

- common backbone = likely basic requirements,
- divergent approaches = legitimate modeling choices,
- conditions explaining different model choices,
- common validation patterns,
- recurring writing structures,
- unique ideas that may be bonus rather than baseline.

Never rank papers by prestige or declare one universally best.

## P3 — Cross-problem synthesis

Create reusable patterns under:

`workspace/knowledge/reusable_patterns/`

Organize by structural archetype rather than historical question letter, e.g.:

- mechanism -> parameter estimation -> optimization,
- baseline -> controlled improvement -> comparison,
- signal/image -> object representation -> geometry/physics -> decision,
- multi-source -> alignment -> fusion -> prediction -> decision,
- source domain -> representation -> adaptation -> interpretation,
- abstract concept -> measurable proxies -> index -> validation,
- graph/resource constraints -> feasible schedule -> multi-objective optimization,
- uncertainty -> probability/robust optimization -> decision.

## P4 — Writing grammar synthesis

Create:

- `workspace/knowledge/writing_patterns/abstract_patterns.md`
- `workspace/knowledge/writing_patterns/question_section_patterns.md`
- `workspace/knowledge/writing_patterns/figure_patterns.md`
- `workspace/knowledge/writing_patterns/validation_patterns.md`

Learn organization, not phrases.

## P5 — Coverage gate

Before declaring historical learning complete, run:

`python scripts/check_pattern_coverage.py`

If any corpus paper lacks a complete card, mark learning status `INCOMPLETE`.

---

# Part C — COMPETITION_SOLVING workflow

When the 2026 problem arrives, do not reread the entire historical corpus first.
Use the synthesized library, then drill into only the most structurally relevant cases.

## Stage 0 — Workspace isolation

Historical practice artifacts must not contaminate the new problem.
Archive or isolate old practice work before importing the 2026 statement/data.

Use `scripts/archive_practice_and_reset.ps1` when ready.

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
- unknown variables/decisions,
- constraints,
- evaluation target,
- required data,
- dependency on previous questions,
- uncertainty/ambiguity.

Do not select a final model yet.

## Stage 1.2 — Historical pattern retrieval

Load `.agents/skills/historical-pattern-retrieval/SKILL.md`.

Retrieve structurally similar historical patterns using dimensions such as:

- task type,
- data modality,
- supervision,
- temporal/spatial structure,
- source/target domain shift,
- uncertainty,
- physical constraints,
- graph/resource constraints,
- final decision type,
- question dependency graph.

Create:

`workspace/analysis/historical_transfer_plan.md`

The plan must include both:

- potentially useful analogies,
- historical analogies explicitly rejected as misleading.

## Stage 1.5 — Solution architecture

Load `.agents/skills/solution-architecture/SKILL.md`.

Create:

- `workspace/analysis/solution_story.md`
- `workspace/analysis/question_dependency_map.md`
- `workspace/analysis/intermediate_representations.md`

Before choosing algorithms, decide what mathematical object flows between questions.
Examples:

- raw signal -> features -> state estimate -> decision,
- image -> object geometry -> spatial relation -> 3D model -> decision,
- mechanism -> latent quantity -> predictor -> optimizer,
- observations -> risk field -> forecast -> route/resource allocation.

A strong solution should read as one system rather than unrelated models pasted together.

## Stage 2 — Data audit / EDA

Load `.agents/skills/data-analysis/SKILL.md`.

Treat `workspace/data/raw/` as immutable.

Audit:

- files and formats,
- dimensions and schemas,
- units,
- missingness,
- duplicates,
- impossible values,
- time/space ordering,
- leakage,
- train/test/domain structure,
- alignment,
- label quality,
- metadata uncertainty.

End with exactly one readiness state:

- `READY_FOR_PILOT`
- `READY_WITH_LIMITATIONS`
- `BLOCKED`

## Stage 2.5 — Evidence recovery when required

If critical metadata are unresolved, load `.agents/skills/evidence-recovery/SKILL.md`.
Do not guess field semantics, units, time zones, QC codes, coordinate systems, or vertical datums.

## Stage 3 — Candidate model design

Load `.agents/skills/model-design/SKILL.md`.

For each key modeling stage, use the pattern:

`baseline -> justified candidate(s) -> falsifiable pilot`

Every model card must answer:

1. What current-problem structure makes this model appropriate?
2. What mathematical representation does it consume and produce?
3. Which assumptions are necessary?
4. What is the simplest credible baseline?
5. What improvement is hypothesized over the baseline?
6. What pilot can falsify this hypothesis quickly?
7. What historical analogy, if any, informed the idea?
8. Why is that analogy transferable here?
9. What parts of the historical case are explicitly not transferred?

Model sophistication is never a selection criterion by itself.

## Stage 4 — Pilot experiments

Load `.agents/skills/experiment/SKILL.md`.

Run the smallest experiments that can distinguish candidate approaches.
Save machine-readable evidence and failure logs.

## Stage 5 — Reviewer gate

Load `.agents/skills/reviewer/SKILL.md`.

Reviewer may reject a model even if it scores well if:

- it does not answer the real question,
- it leaks information,
- its assumptions are unsupported,
- its improvement is not compared fairly,
- it is an unjustified historical transplant,
- it breaks downstream question dependencies,
- its result cannot be reproduced.

## Stage 6 — Full experiments

Scale only reviewer-approved pilots.

## Stage 7 — Validation / sensitivity / robustness / uncertainty

Select checks appropriate to the problem. Do not mechanically run every method.
Potential checks include:

- holdout / cross-validation,
- out-of-distribution/domain-shift tests,
- sensitivity to weights/parameters,
- perturbation/noise robustness,
- ablation,
- uncertainty intervals,
- Monte Carlo propagation,
- feasibility checks,
- physical consistency,
- stability across regions/time periods/subgroups,
- error-case analysis.

## Stage 8 — Visualization

Load `.agents/skills/visualization/SKILL.md`.

Every important figure needs a declared analytical role.
Generate it from saved evidence, not manually entered numbers.

## Stage 9 — Paper drafting

Load `.agents/skills/paper-writing/SKILL.md`.

For each sub-question, prefer the narrative:

`problem-specific difficulty -> abstraction -> model rationale -> formulation -> solution -> result -> validation -> direct answer -> handoff to next question`

Do not write textbook-style algorithm introductions unless a short explanation is necessary for the derivation.

## Stage 10 — Final audit

Load `.agents/skills/final-audit/SKILL.md`.

Audit consistency across:

- problem statement,
- assumptions,
- symbols,
- equations,
- code,
- inputs,
- outputs,
- tables,
- figures,
- abstract,
- conclusions,
- references.

---

# Part D — Writing principles learned from excellent-paper patterns

The following are recurring patterns observed in the historical corpus and should be treated as heuristics, not mandatory templates.

## D1. Abstract

A strong competition abstract usually compresses the full solution chain:

`core problem -> Q1 method + result -> Q2 method + result -> ... -> validation/robustness -> practical conclusion`

Use concrete verified results when available.
Avoid long generic background paragraphs.

## D2. Question sections

A strong section often follows:

1. problem analysis,
2. modeling rationale,
3. mathematical formulation,
4. solver/algorithm,
5. results,
6. validation/discussion,
7. direct answer / mini-summary.

## D3. Progressive questions

When later questions extend earlier ones, preserve the progression explicitly.
Do not restart from zero unless the problem genuinely requires it.

## D4. Baseline and controlled improvement

A baseline is valuable when it gives the reader a reference point.
Improvements are strongest when one can explain exactly what weakness is being fixed.

## D5. Representation is often the real modeling contribution

Before choosing a solver, identify the useful representation:

- a graph,
- a latent state,
- an equivalent physical quantity,
- a geometry parameter vector,
- a risk field,
- a feature space,
- a probability distribution,
- a multi-objective cost.

## D6. Validation should match the claim

Prediction accuracy does not validate a physical interpretation.
A pretty 3D plot does not validate spatial connectivity.
An optimizer returning a value does not prove feasibility or robustness.
Choose validation according to the claim being made.

---

# Part E — Coding and evidence rules

- Use relative paths.
- Never overwrite raw data.
- Fix random seeds where relevant.
- Save result tables to `workspace/results/`.
- Save figures to `workspace/figures/`.
- Save experiment records to `workspace/experiments/`.
- Fail loudly on missing critical inputs.
- Do not hide failed runs.
- Every important result should be regenerable from scripts and source data.

---

# Default action for a brand-new 2026 problem

1. Read this file.
2. Read the 2026 problem and all attachments.
3. Run Stage 1 only.
4. Run Stage 1.2 historical pattern retrieval.
5. Run Stage 1.5 solution architecture.
6. Stop and review architecture before committing to full modeling.
