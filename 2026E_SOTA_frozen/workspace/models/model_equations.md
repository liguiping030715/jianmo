# Stage 3 Model Equations

## Domain and restrictions

Let

- `S={A,B}` be the two stations, processed independently;
- `T={1,...,6}` index the UTC observation-end times `00:00, 00:06, ..., 00:30`;
- `K={1,...,33}` index the ordered common gate tokens listed in `pilot_scope_decision.md`.

The mapping `k -> sampling_gate_token` is categorical and order-preserving only. It is not a mapping to metres, kilometres, AGL, or MSL.

## Resolved input variables

For each `(s,t,k)`:

`d=W1 [degree]`, `V=W2 [m/s]`, `w=W3 [m/s]`, `c^H=W4 [%]`, `c^V=W5 [%]`.

Define

`theta = pi d / 180`,

`x = V cos(theta)`, `y = V sin(theta)`,

`q^H=c^H/100`, `q^V=c^V/100`.

The ordered pair `(x,y)` is a circular embedding, not a claim about east/north axes. For any common angular shift `phi`, rotating every `(x,y)` by `phi` preserves all Euclidean norms and differences used below. Replacing `d` by `d+360n` also leaves the equations unchanged.

## Confidence convention

Confidence is used monotonically as a reliability weight:

`0 <= q^H,q^V <= 1`.

No undocumented threshold is allowed. Structural missingness is handled by the predeclared 33-gate balanced scope, not by confidence. The unweighted ablation sets `q^H=q^V=1` without changing the raw data.

## Candidate 0

`P^(0)_{s,t,k} = [q^H V^2 + q^V w^2]^(1/2)`.

Properties:

- unit: m/s;
- nonnegative;
- no circular-angle issue because `d` is unused;
- no fitted parameter;
- no time or gate interaction.

## Candidate 1

For two cells `a,b` from one station, define

`Q^H(a,b)=min(q^H_a,q^H_b)`,

`Q^V(a,b)=min(q^V_a,q^V_b)`,

`D_beta(a,b) = {Q^H[(x_a-x_b)^2+(y_a-y_b)^2] + beta^2 Q^V(w_a-w_b)^2}^(1/2)`.

Temporal neighbor sets are

- `N_T(1)={2}`;
- `N_T(t)={t-1,t+1}` for `2<=t<=5`;
- `N_T(6)={5}`.

Gate neighbor sets are

- `N_K(1)={2}`;
- `N_K(k)={k-1,k+1}` for `2<=k<=32`;
- `N_K(33)={32}`.

Then

`T_{s,t,k}=median_{r in N_T(t)} D_beta((t,k),(r,k))`,

`G_{s,t,k}=median_{j in N_K(k)} D_beta((t,k),(t,j))`,

`P^(1)_{s,t,k}=[alpha T^2+(1-alpha)G^2]^(1/2)`.

Constraints: `0<=alpha<=1`, `beta>0`. The raw form has m/s units. No term is divided by elapsed time or gate-token spacing.

## Candidate 2

### Robust centering and scaling

For `c in {x,y,w}`, define the gatewise temporal center

`m_{s,k,c}=median_t X_{s,t,k,c}`,

`E_{s,t,k,c}=X_{s,t,k,c}-m_{s,k,c}`.

Define a station-component robust scale over all permitted `(t,k)` residuals:

`sigma_{s,c}=1.4826 median_{t,k}|E_{s,t,k,c}-median_{t,k}E_{s,t,k,c}|`.

If `sigma_{s,c}` is zero or nonfinite, execution must fail for this candidate and record the reason. Otherwise

`Z_{s,t,k,c}=E_{s,t,k,c}/sigma_{s,c}`.

### Weighted low-rank objective

Let `Q_x=Q_y=q^H` and `Q_w=q^V`. For rank `r<=2`, independently for each station:

`J(A,B)=sum_{t,k,c} Q_{t,k,c}[Z_{t,k,c}-sum_{l=1}^r A_{t,l}B_{k,c,l}]^2 + lambda(||A||_F^2+||B||_F^2)`.

Estimate

`(A_hat,B_hat)=argmin J(A,B)`

by deterministic alternating weighted least squares initialized from the unweighted SVD. Factor non-uniqueness is acceptable because only the reconstruction and residual are interpreted.

`R_{t,k,c}=Z_{t,k,c}-sum_l A_hat_{t,l}B_hat_{k,c,l}`,

`P^(2)_{s,t,k}=[q^H(R_x^2+R_y^2)+q^V R_w^2]^(1/2)`.

The output is dimensionless and descriptive.

## Normalization-sensitivity variants

Normalization is never chosen by apparent accuracy. The pilot compares:

1. native-unit formulation where applicable;
2. station-component robust scaling using the declared median/MAD rule;
3. an unweighted confidence ablation.

Any zero scale is an explicit failure, not silently replaced. Candidate 2 requires robust scaling; Candidates 0 and 1 use it only as a sensitivity variant.

## Perturbation-stability quantities

For full-data output `P` and a perturbation output `P^(-j)` evaluated on common cells, define the robust normalized change

`Delta_j = median_common |P^(-j)-P| / max(IQR_common(P), machine_scale)`,

where

`machine_scale = eps_float64 max(1, median_common |P|)`.

`machine_scale` is only a numerical comparison floor; it is not added to a candidate score. Report also Spearman rank correlation on common cells when neither output is constant.

Leave-one-time removal deletes one complete station-time profile and refits/recomputes the candidate. Leave-one-gate removal deletes the same gate token from every time in a station. No omitted value is predicted and no perturbation statistic is called accuracy.

## Dimensional audit

| Quantity | Dimension |
|---|---|
| `V,w,x,y,D,T,G,P^(0),P^(1)` raw form | m/s |
| `q^H,q^V,alpha,beta` | dimensionless (`beta` is a component multiplier) |
| `Z,R,P^(2)` | dimensionless |
| gate index/token | categorical/ordinal only |

No equation contains physical height, height spacing, a vertical derivative, microwave data, or `W6`.
