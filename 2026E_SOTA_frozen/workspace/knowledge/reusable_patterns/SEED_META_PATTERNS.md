# Huawei Cup Historical Meta-Patterns — Seed Library

This file is a **seed**, not a substitute for full-corpus mining.
It records structural patterns already visible across representative 2024-2025 excellent papers.

## Pattern 1 — Mechanism/latent quantity -> estimation -> optimization

Observed in problems where the final decision depends on a quantity that is not directly usable at first.
Typical flow:

`raw observations -> physically/statistically meaningful latent quantity -> predictive/estimation model -> constrained optimization -> robustness`

Use when the problem contains a meaningful mechanism and a downstream decision.

## Pattern 2 — Baseline -> diagnose weakness -> controlled improvement

A strong improvement story is:

`baseline -> identify specific failure mode -> introduce one justified improvement -> compare under same metric/split -> retain only demonstrated gains`

This is stronger than presenting a complex model without a reference point.

## Pattern 3 — Raw signal/image -> object-level representation -> higher-level model

Typical flow:

`pixels/signal -> denoised/segmented structure -> object/geometric/feature representation -> spatial/physical/probabilistic model -> decision`

The intermediate representation is often more important than the classifier itself.

## Pattern 4 — Multi-source -> alignment -> fusion -> prediction/reconstruction -> decision

Do not fuse heterogeneous sources before resolving time, space, units, reliability, and missingness.

## Pattern 5 — Domain shift -> representation -> adaptation -> interpretation

When training/source and target/operational distributions differ, first demonstrate the shift, then justify adaptation. Do not invoke transfer learning merely because two datasets exist.

## Pattern 6 — Abstract concept -> measurable proxies -> composite index -> external validation

For concepts such as quality, aesthetics, risk, resilience, or experience:

`semantic concept -> observable proxies -> normalized indicators -> weighting/aggregation -> sensitivity/external validity`

The key contribution is often proxy design, not the weighting formula.

## Pattern 7 — Resource/graph constraints -> feasible baseline -> multiobjective tradeoff

For scheduling, routing, allocation, or graph problems:

`formal feasibility constraints -> simple constructive baseline -> improvement heuristic/optimizer -> Pareto/tradeoff analysis -> runtime/feasibility validation`

## Pattern 8 — Uncertainty should propagate to the final decision

If early-stage estimates are uncertain and later decisions depend on them, propagate uncertainty rather than presenting a deterministic downstream answer with false precision.

## Pattern 9 — Progressive question chain

Excellent solutions often reuse earlier outputs explicitly. A later question should usually build on a defined prior representation, model, or estimate rather than starting an unrelated pipeline.

## Pattern 10 — Validation must match the claim

- prediction claim -> predictive validation,
- physical-mechanism claim -> physical consistency / residual structure,
- optimization claim -> feasibility + objective comparison + stability,
- reconstruction claim -> geometry/error/uncertainty validation,
- ranking claim -> weight sensitivity / external consistency,
- generalization claim -> held-out domain or perturbation tests.
