# Stage 3 Candidate Eligibility Decision

## Decision

Advance Candidates 0, 1, and 2 to the **restricted component pilot** defined in `workspace/experiments/pilot_plan.md`. This is eligibility to test numerical behavior, not model selection or endorsement.

## Eligible candidates

### Candidate 0 — eligible as negative-control baseline

It is the required simple baseline, uses only resolved units, and provides a deterministic reference for scale domination and pipeline correctness. It is not eligible to be interpreted as turbulence intensity or promoted as a substantive final method merely because it is stable.

### Candidate 1 — eligible

It is the simplest candidate that represents local temporal and ordinal-gate variability, handles direction circularly, uses confidence as documented reliability, and requires no fitting. Its main pilot question is sensitivity to neighbor deletion, normalization, `alpha`, and `beta`.

### Candidate 2 — eligible with strict safeguards

Its rank is capped at two, stations are fit independently, and the factorization is descriptive. It is eligible only with deterministic initialization, fail-loud robust scaling, convergence logging, and the predeclared perturbation checks. Six times are insufficient for performance claims or model selection.

## Candidates rejected before pilot

None of Candidates 0-2 is rejected before pilot because each has a distinct, bounded diagnostic role and obeys the permitted data scope.

The following **design variants are rejected before pilot**:

1. raw-degree subtraction or ordinary averaging of `W1`;
2. conversion of gate tokens to physical height or use of vertical derivatives;
3. any model using microwave data, model `a`, `W6`, radial velocity, spectrum width, or SNR;
4. neural networks, tree ensembles, high-order regressions, or other high-capacity learners on six times;
5. random cell/gate train-test splits;
6. station pooling that removes station identity;
7. confidence thresholds not defined by authoritative metadata;
8. any candidate whose success criterion is accuracy, prediction skill, or agreement with an unavailable reference.

## Comparative pilot priority

Pilot order, without implying preference as a final model:

1. Candidate 0 to verify the data-to-output path and expose raw scale structure.
2. Candidate 1 to test the minimal local-variability mechanism.
3. Candidate 2 only after the first two pass scope and numerical checks, because it introduces fitted factors and normalization dependence.

## Promotion rule after the future pilot

A candidate may survive the component pilot only if it passes the finite-output, circular-invariance, normalization, leave-one-time, leave-one-gate, station-wise, and confidence-ablation checks. Survival would authorize further review, not selection as the final competition model.

## Stage boundary

No pilot has been run. No numerical result, accuracy estimate, generalization claim, final turbulence indicator, or final competition model has been produced. Model `a` and the full Question 1 workflow remain blocked.

