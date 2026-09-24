# Model Card: Candidate 2 — Confidence-Weighted Multivariate Low-Rank Residual

## Role

Small descriptive multivariate candidate testing whether coherent wind structure can be separated from localized residuals without a high-capacity learner.

## Inputs and scope

One fit per station; six permitted UTC times; 33 ordered gate tokens; `W1`-`W5` only.

## Preprocessing and variables

Form circular components `(x,y)` from `W1/W2` and use `w=W3`. Subtract each gate's six-time component median and divide by a station-component MAD scale. Zero or nonfinite scale is a fail-loud condition. `W4/W5` become horizontal/vertical reliability weights.

## Objective and output

Fit a confidence-weighted rank-`r` factorization, `r<=2`, of standardized component values over time and gate rank with optional ridge `lambda`. Use deterministic weighted alternating least squares initialized by unweighted SVD.

Output `P^(2)=sqrt(q^H(R_x^2+R_y^2)+q^V R_w^2)`, a nonnegative dimensionless residual score.

## Ordered gate and time treatment

Gates have indexed loadings but no physical coordinates. Six time factors describe the observed array; they are not a forecasting model. Leave-one-time checks refit on five times and compare retained cells only.

## Parameters

`r in {1,2}`, `lambda in {0,10^-4,10^-2}`, fixed robust scaling, relative-objective tolerance `10^-8`, and iteration cap `500`. None is selected by accuracy.

## Assumptions

Coherent background structure is low rank; residual energy is a useful diagnostic; robust scaling prevents unit/scale domination; confidence is a monotone reliability weight.

## Cost

`O(I S T K C r^2)` for a small iteration count, `T=6`, `K=33`, `C=3`, `r<=2`.

## Strengths

Uses all resolved field semantics, separates component scales, captures coherent multivariate structure, and remains small and deterministic.

## Failure modes

Overfit or non-identifiability with six times; rank/ridge/normalization sensitivity; zero MAD; solver nonconvergence; one-component domination; organized flow residuals mistaken for activity; constant confidence prevents weighting assessment.

## Pilot and rejection criteria

Run deterministic convergence, objective monotonicity, circular invariance, rank/ridge, normalization, confidence ablation, leave-one-time/gate, and station-wise checks. Reject on any numerical failure, material initialization dependence, persistent scale domination, extreme perturbation sensitivity, non-interpretable output, or scope violation.
