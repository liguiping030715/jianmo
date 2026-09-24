---
name: model-design
description: Generate and compare mathematically justified candidate models after problem analysis, SOTA literature scan, historical retrieval, negative-transfer audit, solution architecture, and data readiness checks.
---

# Model Design

## Required inputs

Read:

- problem breakdown and task fingerprint,
- SOTA literature scan and method matrix,
- historical transfer plan,
- negative-transfer audit,
- solution story and intermediate representations,
- validation blueprint,
- data audit/readiness.

Do not design around an unavailable field, unverified mask meaning, external dataset, or unsupported assumption.

## Candidate design

For each key modeling stage, create a simple credible baseline and normally no more than 1–2 serious alternatives unless evidence justifies more.

For every candidate document:

- current-problem rationale,
- input representation,
- output representation,
- assumptions,
- equations/objective,
- constraints,
- parameters,
- solver/training procedure,
- diagnosed baseline weakness,
- hypothesized improvement,
- current-data evidence supporting the mechanism,
- recent literature support if any,
- important mismatch from the literature setting,
- historical analogy if any,
- why that analogy transfers,
- what historical details do not transfer,
- external-data requirements,
- compute/time/dependency cost,
- failure modes,
- pilot experiment,
- rejection criteria,
- downstream compatibility.

## Literature discipline

A citation does not justify a model by itself.
A top-venue method remains only a candidate until the current data and a fair pilot support it.

Do not use external training data when the official problem prohibits it.
A pretrained tool/model may be used only within the official rules and must be documented.

## Selection principle

Prefer the simplest model that adequately answers the problem and survives validation.
Complexity is justified only by:

1. a diagnosed limitation of a simpler baseline, or
2. a structural requirement of the task,

and must survive a falsifiable pilot.
