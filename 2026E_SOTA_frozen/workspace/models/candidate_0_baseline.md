# Model Card: Candidate 0 — Confidence-Weighted Magnitude Baseline

## Role

Negative-control baseline for the restricted profiler-only component pilot. It is not a turbulence model.

## Inputs and scope

Stations A/B separately; six permitted UTC times; 33 ordered gate tokens; `W2` horizontal speed, `W3` vertical speed, and `W4/W5` confidence. `W1` is intentionally unused.

## Equation

With `q^H=W4/100` and `q^V=W5/100`:

`P^(0)=sqrt(q^H W2^2 + q^V W3^2)`.

Output is nonnegative and has m/s units.

## Assumptions and constraints

The score is useful only as a simple magnitude reference. It does not assume that mean wind magnitude equals turbulence. Gates are independent categorical indices; times are independent observations. No physical height, microwave field, `W6`, interpolation, or fitted parameter is used.

## Confidence and circular direction

`W4/W5` are reliability weights without thresholds. `W1` is unused, so no circular transformation is needed.

## Solver and cost

Closed form, `O(S T K)` time and `O(T K)` memory per station.

## Strengths

Transparent, reproducible, dimensionally valid, and useful for detecting whether complex candidates merely reproduce wind-speed scale.

## Failure modes

Dominated by horizontal mean flow; no variability mechanism; no time/gate interaction; confidence dependence may be unidentifiable.

## Pilot and rejection criteria

Run finite/nonnegative/reproducibility, normalization, contribution, time/gate, leave-one-time/gate, station, and confidence-ablation checks. Reject the implementation on NaN/Inf, nondeterminism, scope violation, or changes on retained cells after deletion. Do not reject its baseline role merely for horizontal-speed dominance.

