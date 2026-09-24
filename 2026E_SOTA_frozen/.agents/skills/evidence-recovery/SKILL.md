---
name: evidence-recovery
description: Resolve missing data dictionaries, field semantics, units, time conventions, coordinate systems, QC meanings, and other metadata blockers using authoritative evidence before model design.
---

# Evidence Recovery Skill

## Purpose

Use this skill when Stage 2 data analysis discovers unresolved metadata that would make later modeling unsafe.

Typical triggers include:

- unknown column meanings
- unknown units
- undocumented category or profile codes
- unclear QC flags
- unclear missing-value conventions
- unknown coordinate reference system
- unclear vertical datum
- unclear time zone or timestamp semantics
- inconsistent station metadata
- missing official data dictionary

This stage exists to recover evidence, not to invent interpretations.

---

# Inputs

Read:

- `workspace/analysis/problem_breakdown.md`
- `workspace/analysis/variables_and_constraints.md`
- `workspace/analysis/assumptions.md`
- `workspace/analysis/data_inventory.md`
- `workspace/analysis/data_quality_report.md`
- `workspace/analysis/time_height_alignment.md`
- `workspace/analysis/stage2_readiness.md`

Also inspect the original files under:

- `workspace/problem/`
- `workspace/data/raw/`

---

# Evidence hierarchy

Classify every external source:

## Level A

Official competition documentation, government standards, official technical standards, official data dictionaries.

## Level B

Equipment manufacturer manuals, official software documentation, peer-reviewed technical papers that explicitly define the same data format.

## Level C

University notes, technical blogs, repositories, secondary documentation.

## Level D

Inference based only on values, naming, magnitude, or common practice.

Only Level A or Level B evidence may change a field from `UNKNOWN` to `RESOLVED`.

Level C may support investigation but cannot independently resolve a critical field.

Level D must remain an inference.

---

# Required workflow

## Step 1: Build blocker register

Extract every unresolved item from Stage 2.

For each blocker record:

- identifier
- file
- field
- current status
- why it blocks modeling
- required evidence

## Step 2: Search authoritative documentation

Prefer sources in this order:

1. competition organizer documentation
2. official national / industry standards
3. equipment manufacturer documentation
4. official institutional technical documentation
5. peer-reviewed literature
6. secondary sources

Never use a secondary source when an authoritative source is available.

## Step 3: Verify exact schema match

Do not assume that a similar instrument uses the same format.

Before resolving a field, verify where possible:

- product name
- file format
- field order
- number of columns
- units
- version
- timestamp structure
- station metadata structure

## Step 4: Record provenance

For every resolved field record:

- raw field name
- resolved meaning
- unit
- evidence source
- authority level
- confidence
- exact reason it matches the supplied file format

## Step 5: Preserve unresolved items

If authoritative evidence cannot be found:

mark the item as:

`UNRESOLVED`

Do not guess.

---

# Prohibited behavior

Never:

- infer field meaning only from numerical magnitude
- invent units
- invent QC meanings
- silently reinterpret timestamps
- assume AGL/MSL without documentation
- assume duplicate records are errors
- delete or modify raw data
- use an online solution's model result as evidence
- start model design while critical schema blockers remain

---

# Required outputs

Create:

`workspace/analysis/schema_resolution.md`

`workspace/analysis/evidence_register.md`

`workspace/analysis/stage2_5_readiness.md`

Do not overwrite previous Stage 2 reports.

---

# Readiness decision

At the end choose exactly one:

## READY_FOR_PILOT

All critical metadata required for a pilot experiment is supported by authoritative evidence.

## READY_WITH_LIMITATIONS

A valid pilot can be run on a clearly defined subset, but some stations, fields, periods, or variables must remain excluded.

The excluded scope must be explicitly documented.

## BLOCKED

Critical unresolved metadata prevents a scientifically defensible pilot.

---

# Handoff rule

If:

`READY_FOR_PILOT`

or

`READY_WITH_LIMITATIONS`

then the next allowed stage is:

`model-design`

If:

`BLOCKED`

do not continue to model design.

Return the unresolved blocker list instead.