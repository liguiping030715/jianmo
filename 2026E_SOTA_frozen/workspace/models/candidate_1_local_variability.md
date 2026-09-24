# Model Card: Candidate 1 — Local Time-Gate Circular Variability

## Role

Closed-form local variability candidate for the restricted profiler-only component pilot.

## Inputs and scope

Stations separate; six permitted UTC times; 33 ordered gate tokens; `W1`-`W5` only.

## Core construction

Embed direction and speed as `x=V cos(theta)`, `y=V sin(theta)` with `theta=pi W1/180`. Define confidence-weighted Euclidean differences of `(x,y,w)` between adjacent times at the same gate and adjacent gate ranks at the same time. Do not divide by time or gate spacing.

`P^(1)=sqrt(alpha T^2+(1-alpha)G^2)`, where `T` and `G` are median adjacent temporal and gate-rank differences. Anchor parameters are `alpha=0.5`, `beta=1`; all sensitivity values are predeclared in `pilot_plan.md`.

## Confidence and circular direction

`W4/W5` are pairwise reliability weights through the minimum confidence of the two cells. They are not masks. Circular embedding prevents a false discontinuity at 0/360 degrees.

## Ordered gate and time treatment

Only rank adjacency is used. Endpoints have one neighbor and interior cells have two. The six times support lag-one changes only. No physical height or longer time-scale inference is made.

## Output

Temporal component `T`, gate component `G`, and combined nonnegative score `P^(1)`, in m/s for native-unit form.

## Assumptions

Local changes can serve as an activity diagnostic; ordinal adjacency is meaningful without being a physical distance; confidence is a monotone reliability weight.

## Solver and cost

Closed form, `O(S T K)`.

## Strengths

Interpretable, circularly valid, no fitting, and explicitly separates time/gate contributions.

## Failure modes

Deletion changes neighborhoods; unequal unknown physical gate spacing; sensitivity to `alpha`, `beta`, and normalization; noise or organized flow can appear as high activity; constant confidence prevents empirical weighting assessment.

## Pilot and rejection criteria

Run all common diagnostics plus parameter grid and separate `T/G` inspection. Reject on NaN/Inf, circular-invariance failure, persistent scale domination, arbitrary-normalization dependence, non-interpretable structure, extreme leave-one-time/gate instability, or any forbidden data/height use.

