# Stage 3 Candidate Models: Restricted Profiler-Only Component

## Scope lock

These candidates are designs for a component-level pilot of profiler-only model `b`. They are not a full Question 1 solution and are not estimates of a validated turbulence quantity.

Every candidate is restricted to:

- station A and station B processed independently;
- the six confirmed UTC observation-end times;
- the 33 common, fully numeric sampling-gate tokens, treated only as ordered indices;
- resolved fields `W1`-`W5`;
- no microwave data, model `a`, `W6`, physical-height conversion, physical-height derivative, or height-based interpolation.

The common output symbol `P^{(m)}_{s,t,k}` means a **candidate profiler activity score** for candidate `m`. It must not be called turbulence intensity, accuracy, or ground truth.

## Common notation

For station `s`, time index `t=1,...,6`, and ordered gate index `k=1,...,33`:

- `d_{s,t,k}`: `W1`, horizontal wind direction in degrees;
- `V_{s,t,k}`: `W2`, horizontal wind speed in m/s;
- `w_{s,t,k}`: `W3`, vertical wind speed in m/s, downward positive;
- `c^H_{s,t,k}`: `W4`, horizontal confidence in percent;
- `c^V_{s,t,k}`: `W5`, vertical confidence in percent;
- `q^H=c^H/100`, `q^V=c^V/100`.

When direction is used, define `theta=pi*d/180` and the coordinate-neutral circular embedding

`x=V cos(theta)`, `y=V sin(theta)`.

`x` and `y` are not labeled eastward/northward components because the supplied schema does not document a from/to or axis convention. Their Euclidean differences are invariant to adding 360 degrees and to a common rotation or sign reversal.

## Candidate 0: confidence-weighted magnitude baseline

### 1. Purpose

Provide the simplest deterministic negative-control baseline. It checks parsing, units, confidence weighting, and output plumbing. It deliberately measures instantaneous wind magnitude rather than variability, so it establishes how much apparent structure can arise without a turbulence-specific mechanism.

### 2. Exact permitted inputs

`W2`, `W3`, `W4`, and `W5` on the permitted `(s,t,k)` cells. `W1` is retained in the input table but is intentionally unused.

### 3. Preprocessing

Convert confidence percentages to `[0,1]` weights. Do not center, interpolate, impute, or convert gate tokens. Base form uses native wind units. A robust-scaled form is evaluated only as a normalization-sensitivity variant.

### 4. Mathematical notation

`q^H`, `q^V`, `V`, and `w` are as defined above.

### 5. Equations

`P^(0)_{s,t,k} = sqrt(q^H_{s,t,k} V_{s,t,k}^2 + q^V_{s,t,k} w_{s,t,k}^2)`.

The objective is evaluation in closed form; there is no fitted loss or solver.

### 6. How `W1`-`W5` are used

- `W1`: not used, by design.
- `W2`: horizontal-magnitude term.
- `W3`: vertical-magnitude term.
- `W4`: reliability weight on `W2`.
- `W5`: reliability weight on `W3`.

### 7. Role of `W4`/`W5`

Reliability weights, not masks and not learned features. No confidence threshold is introduced.

### 8. Circular wind direction

Not applicable because `W1` is unused. This avoids a circularity error in the baseline.

### 9. Ordered gate dimension

Each gate is evaluated independently. Gate ordering is preserved in the output but not used numerically.

### 10. Six time points

Each time is evaluated independently. No temporal fitting or train/test split occurs.

### 11. Output definition

A nonnegative score in m/s with shape `2 stations x 6 times x 33 gates`. It is an instantaneous confidence-weighted speed magnitude, not turbulence intensity.

### 12. Parameters

No fitted parameter. The confidence mapping is fixed by the documented percent scale.

### 13. Assumptions

Only that a simple wind-magnitude field is useful as a negative-control reference. It does not assume that stronger mean wind implies stronger turbulence.

### 14. Computational cost

`O(S T K)` time and `O(T K)` working memory per station; here only 396 permitted cells in total.

### 15. Expected strengths

Transparent, unit-consistent, finite for valid inputs, reproducible, and useful for detecting scale domination in more elaborate candidates.

### 16. Failure modes

Dominance by horizontal speed; inability to distinguish steady flow from variability; weak vertical-velocity contribution; no time/gate interaction; confidence effects unidentifiable when confidence is constant.

### 17. Scientific defensibility under the restricted scope

It uses only resolved quantities and makes a deliberately narrow claim. Its defensible role is baseline plumbing and negative control, not physical turbulence inference.

## Candidate 1: local time-gate circular variability

### 1. Purpose

Measure local changes of the three wind components along two permitted axes: adjacent UTC observations at a fixed gate and adjacent ordered gates at a fixed time. It avoids physical-height derivatives by using ordinal adjacency only.

### 2. Exact permitted inputs

`W1`-`W5` on all permitted cells. Stations remain separate.

### 3. Preprocessing

Convert direction to radians, form the circular embedding `(x,y)`, retain `w`, and convert confidences to `[0,1]`. No physical gate spacing is used. Base form remains in m/s; robust component scaling is a sensitivity variant.

### 4. Mathematical notation

Let `a=(t,k)` and `b=(r,j)` be two cells from the same station. Define pair weights

`Q^H(a,b)=min(q^H_a,q^H_b)` and `Q^V(a,b)=min(q^V_a,q^V_b)`.

### 5. Equations

The confidence-weighted circular wind difference is

`D_beta(a,b)^2 = Q^H(a,b)[(x_a-x_b)^2+(y_a-y_b)^2] + beta^2 Q^V(a,b)(w_a-w_b)^2`.

For temporal-neighbor set `N_T(t)` and ordinal-gate-neighbor set `N_K(k)`:

`T_{s,t,k} = median_{r in N_T(t)} D_beta((t,k),(r,k))`,

`G_{s,t,k} = median_{j in N_K(k)} D_beta((t,k),(t,j))`,

`P^(1)_{s,t,k} = sqrt(alpha T_{s,t,k}^2 + (1-alpha) G_{s,t,k}^2)`.

`N_T` contains the immediately preceding/following available time indices; endpoints have one neighbor. `N_K` is defined analogously over gate rank. This is a closed-form calculation, not a fitted optimization.

### 6. How `W1`-`W5` are used

- `W1` and `W2`: jointly form the circular horizontal-wind embedding.
- `W3`: supplies vertical-wind changes.
- `W4`: weights horizontal differences.
- `W5`: weights vertical differences.

### 7. Role of `W4`/`W5`

Pairwise reliability weights. They are neither hard masks nor predictors. The unweighted result is retained as an ablation.

### 8. Circular wind direction

Direction is never differenced as a raw degree value. The `(cos,sin)` embedding makes 359 degrees and 1 degree close. A `+360 degrees mod 360` invariance test is mandatory.

### 9. Ordered gate dimension

Only immediate predecessor/successor by gate rank is used. All adjacent ranks are treated as adjacency relations, not equal physical distances. No division by a gate-token difference is allowed.

### 10. Six time points

Only adjacent time indices are compared. The six-minute interval is common, but the score is an adjacent-step change, not a time derivative. No lag beyond one step is estimated.

### 11. Output definition

`T`, `G`, and combined nonnegative `P^(1)` at every permitted cell. The raw form has m/s units; a robust-normalized sensitivity form is dimensionless.

### 12. Parameters

- `alpha in [0,1]`: temporal versus ordinal-gate contribution; pilot anchor `alpha=0.5`, with sensitivity at `0`, `0.25`, `0.75`, and `1`.
- `beta>0`: vertical-component multiplier; pilot anchor `beta=1`, with sensitivity at `0.5` and `2`.

These are not fitted to accuracy.

### 13. Assumptions

Local temporal and ordinal-gate changes are useful activity diagnostics; adjacent gate ranks carry neighborhood information even though their physical spacing is unknown; confidence percentages can act monotonically as reliability weights.

### 14. Computational cost

`O(S T K)` because each cell has at most two temporal and two gate neighbors. Memory is `O(T K)` per station.

### 15. Expected strengths

Explicit circular handling, interpretable time and gate contributions, no training, no physical-height dependency, and direct leave-one-time/gate sensitivity checks.

### 16. Failure modes

High sensitivity to one observation; discontinuities when deleting a gate changes its neighbors; ordinal adjacency may mix unequal physical separations; arbitrary balance through `alpha`/`beta`; confidence weighting untestable if all values are 100; local change can include measurement noise or organized shear rather than turbulence.

### 17. Scientific defensibility under the restricted scope

All differences are between resolved wind quantities with compatible units. Gate use is explicitly ordinal, and the output is labeled a local variability score rather than a physical gradient or validated turbulence value.

## Candidate 2: confidence-weighted multivariate low-rank residual

### 1. Purpose

Separate coherent station-specific time-gate structure from localized multivariate departures in horizontal and vertical wind, using a deliberately low-rank descriptive model suited to a `6 x 33` grid.

### 2. Exact permitted inputs

`W1`-`W5` on the permitted cells, with one independent fit per station.

### 3. Preprocessing

Form `(x,y,w)`. For each station and gate, subtract the six-time component median. Scale each component using a station-level robust scale computed from the centered residuals. If a required scale is zero or nonfinite, fail loudly for that component/candidate; do not silently add jitter or drop it. Convert confidences to reliability weights.

### 4. Mathematical notation

Let `Z_{s,t,k,c}`, `c in {x,y,w}`, be the centered, robust-scaled component. Let `Q_{s,t,k,x}=Q_{s,t,k,y}=q^H` and `Q_{s,t,k,w}=q^V`.

### 5. Equations and objective

For each station independently, estimate time factors `A_{t,l}` and gate-component loadings `B_{k,c,l}` by

`min_{A,B} sum_{t,k,c} Q_{t,k,c}[Z_{t,k,c} - sum_{l=1}^r A_{t,l}B_{k,c,l}]^2 + lambda(||A||_F^2+||B||_F^2)`.

The standardized residual is

`R_{t,k,c}=Z_{t,k,c}-sum_l A_{t,l}B_{k,c,l}`,

and the output is

`P^(2)_{s,t,k}=sqrt(q^H(R_x^2+R_y^2)+q^V R_w^2)`.

Use deterministic weighted alternating least squares initialized by the unweighted SVD of the same standardized matrix. Declare convergence when relative objective change is at most `10^-8`; stop and mark nonconvergence after `500` iterations.

### 6. How `W1`-`W5` are used

- `W1`, `W2`: circular horizontal embedding.
- `W3`: third physical wind component.
- `W4`: horizontal residual weight.
- `W5`: vertical residual weight.

### 7. Role of `W4`/`W5`

Reliability weights in both fitting objective and residual score. They are not hard masks or ordinary features. An unweighted ablation is required.

### 8. Circular wind direction

Handled exclusively through `(x,y)`. Rotation and `+360 degrees` invariance checks are required. No raw angular residual is used.

### 9. Ordered gate dimension

Each gate has a separate loading indexed by rank. No numerical gate spacing, height derivative, or interpolation is used. The model may learn coherent patterns across the ordered array but cannot assign physical wavelengths.

### 10. Six time points

The rank is constrained to `r<=2`; no train/test split is made. Leave-one-time removal refits the descriptive factorization on five times only and compares outputs on common cells as a stability test, not prediction.

### 11. Output definition

A nonnegative, dimensionless, confidence-weighted residual score with shape `6 x 33` per station. It measures departure from a low-rank description, not validated turbulence.

### 12. Parameters

- rank `r in {1,2}`;
- ridge `lambda in {0, 10^-4, 10^-2}` for sensitivity only;
- robust scale definition, fixed before execution;
- deterministic ALS relative-objective tolerance `10^-8` and iteration cap `500`.

No parameter is selected using accuracy.

### 13. Assumptions

Coherent background variation is approximately low rank over this tiny time-gate array; localized residual energy is a useful component-pilot diagnostic; robust component scaling is meaningful; confidence percentages provide monotone reliability weights.

### 14. Computational cost

Approximately `O(I S T K C r^2)` for `I` alternating iterations, with `T=6`, `K=33`, `C=3`, and `r<=2`; computationally trivial at this scope.

### 15. Expected strengths

Uses all resolved semantics, separates scale before combining variables, represents coherent multivariate structure, remains deterministic, and exposes whether complexity is supportable with six times.

### 16. Failure modes

Overfitting or unstable factors with only six times; rank/normalization sensitivity; non-identifiability of factors; zero robust scale; residuals dominated by a component; confidence weights empirically untestable when constant; organized non-turbulent departures appearing as high scores.

### 17. Scientific defensibility under the restricted scope

The factorization is descriptive, station-specific, uses no physical height, and makes no accuracy or generalization claim. Its pilot purpose is to test whether a small multivariate residual pipeline is numerically stable enough to merit later investigation.

## Candidate comparison before pilot

| Property | Candidate 0 | Candidate 1 | Candidate 2 |
|---|---|---|---|
| Role | Negative-control baseline | Local variability diagnostic | Multivariate coherent/residual diagnostic |
| Uses direction | No | Circular embedding | Circular embedding |
| Uses time adjacency | No | Yes | Joint low-rank time factors |
| Uses gate order | Output order only | Immediate ordinal neighbors | Gate-indexed loadings |
| `W4/W5` role | Reliability weights | Pairwise reliability weights | Objective and residual weights |
| Fitted parameters | None | None | Low-rank factors only |
| Main risk | Mean wind mistaken for activity | Neighbor/parameter sensitivity | Overfit and normalization sensitivity |
| Pilot status | Eligible as baseline only | Eligible | Eligible with strict rank/scale checks |

No candidate is selected as the final competition model.
