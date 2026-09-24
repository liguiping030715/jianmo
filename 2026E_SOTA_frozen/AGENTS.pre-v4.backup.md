# HuaweiCup Codex — Project Instructions

## Mission

Act as a rigorous mathematical-modeling teammate, not a one-shot answer generator.

Maintain a traceable chain:

problem -> assumptions -> data -> evidence -> model -> code -> results -> paper

---

## Required workflow

1. Problem analysis
2. Data audit / EDA
2.5. Evidence recovery when required
3. Candidate model design
4. Pilot experiments
5. Model selection or combination
6. Full experiments
7. Validation / sensitivity / robustness
8. Visualization
9. Paper drafting
10. Final consistency audit

Do not jump directly from the statement to modeling or the final paper.

---

## Stage 1 — Problem analysis

When a new problem is added, load:

`.agents/skills/problem-analysis/SKILL.md`

Before modeling, create:

- `workspace/analysis/problem_breakdown.md`
- `workspace/analysis/variables_and_constraints.md`
- `workspace/analysis/assumptions.md`

Stage 1 must identify:

- sub-problems
- objectives
- inputs and outputs
- known and unknown variables
- constraints
- dependencies between questions
- assumptions
- ambiguity
- required data

Do not select a final model during Stage 1.

---

## Stage 2 — Data audit / EDA

After Stage 1, load:

`.agents/skills/data-analysis/SKILL.md`

Inspect the supplied raw data before model design.

At minimum document:

- file inventory
- data formats
- dimensions
- field names
- missing values
- duplicates
- outliers
- units
- timestamps
- spatial coordinates
- data quality
- leakage risks
- time / spatial / height alignment
- inconsistencies between files

Treat:

`workspace/data/raw/`

as immutable.

Derived data must go to:

`workspace/data/processed/`

Do not silently interpret unknown fields.

Do not silently remove, fill, interpolate, deduplicate, or transform questionable records.

---

## Stage 2 readiness decision

At the end of Stage 2, issue exactly one decision:

- `READY_FOR_PILOT`
- `READY_WITH_LIMITATIONS`
- `BLOCKED`

The decision and reasons must be written to a Stage 2 readiness report.

---

## Stage 2 -> Stage 2.5 Gate

If Stage 2 discovers unresolved critical metadata, do NOT proceed directly to model design.

Examples include:

- unknown field meanings
- unknown units
- unknown QC codes
- unknown missing-value conventions
- undocumented category or profile codes
- unclear timestamp semantics
- unclear time zone
- unclear coordinate reference system
- unclear vertical datum
- uncertain station metadata
- unresolved schema inconsistencies
- missing official data dictionary

In that case, load:

`.agents/skills/evidence-recovery/SKILL.md`

and execute Stage 2.5.

Do not bypass this gate.

---

## Stage 2.5 — Evidence recovery

The purpose of Stage 2.5 is to resolve metadata blockers using authoritative evidence.

Preferred evidence order:

1. competition organizer documentation
2. official national / industry standards
3. government documentation
4. equipment manufacturer documentation
5. official institutional technical documentation
6. peer-reviewed literature
7. secondary technical sources

Do not resolve critical fields using numerical appearance, intuition, or common practice alone.

Evidence recovery must preserve provenance.

For each resolved field record:

- original field
- resolved meaning
- unit
- evidence source
- authority level
- confidence
- why the source matches the supplied data format

Create:

- `workspace/analysis/schema_resolution.md`
- `workspace/analysis/evidence_register.md`
- `workspace/analysis/stage2_5_readiness.md`

Do not destructively overwrite Stage 2 reports.

---

## Stage 2.5 readiness decision

At the end of Stage 2.5, issue exactly one decision:

- `READY_FOR_PILOT`
- `READY_WITH_LIMITATIONS`
- `BLOCKED`

### READY_FOR_PILOT

Critical metadata needed for a pilot is adequately supported.

### READY_WITH_LIMITATIONS

A valid pilot can be run only on a clearly defined subset.

Any excluded:

- station
- field
- time period
- sensor
- variable
- record group

must be documented.

### BLOCKED

Critical uncertainty still prevents a scientifically defensible pilot.

---

## Stage 2.5 -> Stage 3 Gate

Only proceed to candidate model design when Stage 2 or Stage 2.5 concludes:

- `READY_FOR_PILOT`
- `READY_WITH_LIMITATIONS`

If the current decision is:

`BLOCKED`

stop.

Report unresolved blockers instead of inventing assumptions.

---

## Stage 3 — Candidate model design

Load:

`.agents/skills/model-design/SKILL.md`

Before full experiments, create at least two plausible candidate approaches when meaningful alternatives exist.

For each candidate document:

- modeling idea
- why it fits the problem
- required assumptions
- symbols
- equations
- objective function
- constraints
- parameters
- solver / optimization strategy
- required data
- computational cost
- strengths
- weaknesses
- likely failure modes
- pilot experiment
- rejection criteria

Do not select a model only because it is sophisticated.

Use a baseline when meaningful.

---

## Stage 4 — Pilot experiments

Load:

`.agents/skills/experiment/SKILL.md`

Pilot experiments should be small, reproducible, and designed to eliminate weak approaches early.

For every important experiment save:

- script
- exact command
- input paths
- parameters
- random seed where relevant
- output paths
- metrics
- runtime if meaningful
- warnings
- limitations

Prefer machine-readable outputs such as:

- CSV
- JSON

plus a short Markdown explanation.

---

## Stage 5 — Reviewer

Load:

`.agents/skills/reviewer/SKILL.md`

The reviewer must independently check:

- problem alignment
- assumption validity
- data integrity
- schema interpretation
- mathematics
- leakage
- baselines
- metrics
- constraints
- robustness
- reproducibility
- figure/result consistency
- claim strength

Critical failures block progression.

The reviewer must be allowed to reject:

- a model
- an experiment
- a result
- an assumption
- a paper claim

---

## Stage 6 — Full experiments

Only run full experiments after candidate approaches have passed pilot and reviewer checks.

Do not scale up a failed or unjustified pilot.

Save all important outputs under:

- `workspace/experiments/`
- `workspace/results/`
- `workspace/figures/`

---

## Stage 7 — Validation / sensitivity / robustness

Where relevant evaluate:

- parameter sensitivity
- robustness
- generalization
- uncertainty
- ablation
- baseline comparison
- stability across subsets
- error cases
- failure modes

Do not report only the best-case result.

---

## Stage 8 — Visualization

Figures must be generated from saved results.

Do not manually type numerical values into figures.

Every figure must be traceable to:

- source data
- processing script
- result file

---

## Stage 9 — Paper drafting

Load:

`.agents/skills/paper-writing/SKILL.md`

Do not write polished prose around unverified results.

Before paper writing, every numerical claim must be backed by:

- a saved result
- a saved figure
- a verified external source

Every important claim should be traceable.

---

## Stage 10 — Final consistency audit

Before finalizing the paper, verify consistency between:

- problem statement
- assumptions
- equations
- code
- parameters
- experiment logs
- tables
- figures
- paper text

Check especially for:

- inconsistent symbols
- changed units
- unsupported claims
- numbers that do not match saved results
- missing experiment provenance
- citation mismatches
- conclusions stronger than the evidence

---

## Evidence rules

Never invent:

- data
- references
- metrics
- figures
- field meanings
- units
- QC codes
- experimental results
- model performance

Separate clearly:

- observed fact
- problem-statement fact
- authoritative external evidence
- assumption
- inference
- modeling decision

Unknown information must remain unknown until supported.

---

## Data rules

Treat:

`workspace/data/raw/`

as immutable.

Derived data goes to:

`workspace/data/processed/`

Document:

- missing values
- outliers
- units
- duplicates
- transformations
- leakage risks
- excluded samples
- interpolation
- alignment
- filtering

Never silently alter raw data.

---

## Modeling rules

Do not select a model only because it is sophisticated or popular.

For every selected model define:

- symbols
- assumptions
- equations
- objective
- constraints
- parameters
- solver
- evaluation criteria
- failure modes

Use a baseline when meaningful.

---

## Coding rules

Use relative paths.

Never overwrite raw data.

Save figures and result tables to files.

Fix random seeds where relevant.

Fail loudly on missing inputs.

Do not silently recover from critical parsing or schema errors.

---

## Reviewer role

Reviewer approval is required before promoting pilot results into final results.

Critical failures block progression.

Reviewer findings must not be ignored merely because a model performs well numerically.

---

## Paper rules

Do not write polished prose around unverified results.

Every number in the paper must be traceable to evidence.

Do not describe an inference as an observed fact.

Do not describe a pilot result as a final result.

Do not hide limitations discovered during data audit or evidence recovery.

---

## Competition integrity

Use this workflow only in ways permitted by the current competition rules and organizer requirements.

Do not use external solution repositories as hidden answer sources.

External materials may be used for:

- standards
- documentation
- scientific references
- algorithm background

when permitted by competition rules and when their provenance is recorded.

---

## Default first action

When a new problem is added:

1. Read this `AGENTS.md`.
2. Inspect available skills.
3. Load `problem-analysis` first.
4. Inspect the problem statement and supplied data.
5. Complete Stage 1 before coding.

Do not start coding until the problem decomposition is written.

---

## Global progression rule

Never advance merely because the next stage is available.

Advance only when the previous stage's evidence and readiness gate permit progression.

A `BLOCKED` decision means stop, investigate, and report blockers.

Never bypass a blocked gate by inventing assumptions.