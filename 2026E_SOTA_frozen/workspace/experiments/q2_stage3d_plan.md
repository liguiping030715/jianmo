# Q2 Stage 3D — E1 local restoration and clean/gap consistency

**Frozen before any E1 valid prediction is inspected.** E0 remains
`E0_REJECTED`; R0a, R1 and E0 are read-only frozen comparators.

## Boundary and hypothesis

Only `cleaning_2026e_v2/aligned/{train,valid}` (3395/728 samples), frozen
BERT, existing train-fitted scalers, P/O and the byte-identical
`pilot_masks/aligned_seed{1729,2718,31415}.jsonl` are used. No A2 test,
Attachment 1/3/4 or external sentiment data. Masked text uses the already
frozen post-intervention BERT cache, never clean contextual states.

E0 showed that weighting responded locally but did not produce stable
prediction robustness. E1 asks whether **recovering a compact latent state at
known injected gaps**, then aligning full/gap representations, remedies that
failure. It does not claim to reconstruct original raw features or infer an
official hidden missing mask.

## Architecture, with frozen dimensions

The original R1/B1 `proj_m: d_m→128`, position and tri-state P/O embeddings,
shared `Linear(384,128)+GELU` fusion, classification head and bounded regression
head are retained. For modality `m` and time `t`, define the R1 local state
`z_mt=GELU(proj_m(x_mt)+p_embed(P_mt)+o_embed(O_mt)+pos_mt)`.

For positions in the **known frozen synthetic injected mask** `M_inj` only,
compute a masked mean of same-modality observed `z` at offsets `±1,±2`, and a
masked mean of other modalities' observed `z` at the same index. Context
eligibility is exactly `P=1,O=1`; native unknown `2` is never promoted to
known observation. Concatenate local mean (128), cross-modal mean (128),
local neighbor fraction (1), cross-modal fraction (1), and one-hot target
P/O (3+3), total 264 dimensions. A per-modality
`Linear(264,128)→GELU→Linear(128,128)` estimates `z_hat_mt`.

At `M_inj=1`, use `z_hat` as content for the existing R1 mean pool. At
`O=1`, keep original `z`; at native `O=2`, retain the R1 unknown policy and
**never** force restoration. The original R1 state summary is unchanged and
uses gap-view P/O. The untreated reference under the same E1 weights uses
the ordinary R1 pool that excludes injected O=0 positions.

No D0 task-specific fusion, C1 gate, C2 teacher, Transformer, MoE, diffusion
or raw-feature generative decoder is present.

## Frozen losses and train-only calibration

Each epoch has the same clean and one frozen-gap copy per train sample as R1.
The prediction loss is CE plus the frozen
`lambda_reg=2.0526115894317627` times Huber (delta 1), on each view.
For each injectable gap-view batch, run the **same E1 network** on clean
counterparts. The clean outputs are stop-gradient targets:

`L_local = mean_{M_inj=1} Huber(z_hat_gap, stopgrad(z_full))`,

`L_global = mean_{injectable rows} [1−cos(H_gap, stopgrad(H_full))]`,

`L = L_cls + lambda_reg L_reg + lambda_local L_local + lambda_global L_global`.

No zero-valued row, native unknown state, or A3 suspected gap is target truth.
On clean-only batches both auxiliary losses are zero.

**Calibration happens before any valid forward pass.** With seed 1729's
freshly initialized model and the first 256 injectable train entries in
frozen manifest order, compute train-only mean prediction loss, local loss,
and global loss in batches of 64. Freeze
`lambda_local=clip(0.10*mean_prediction_loss/mean_local_loss,0.01,10)` and
`lambda_global=clip(0.10*mean_prediction_loss/mean_global_loss,0.01,10)`;
denominators below `1e-8` block execution. These values are reused across
all seeds and both ablations. The calibration record includes input hashes,
sample IDs, raw losses and final weights. No valid metric can change them.

## Optimizer, selection and budget

AdamW lr `0.001`, weight decay `0.0001`, head batch `64`, cap `12` epochs,
patience `3`, deterministic seed, CPU/RAM cap `14 GiB`, projected run cap
`2 h`. Checkpoint score is the frozen R1 valid selection loss:
`0.5*clean + 0.5*mean(S1..S6)`, where each term is CE+lambda_reg Huber.
This uses valid for checkpoint selection only; mechanism weights use train only.
Full per-epoch and final audit metrics, predictions, mask manifest and failures
are retained.

## Mechanism gates fixed before training

Use only the six frozen valid **audit** gap masks and eligible paired rows.
Average each scenario's mean over injected positions, then weight scenarios
equally. The full E1 seed 1729 must satisfy both:

1. `restoration_gain = 1−Huber(z_hat,z_full)/Huber(0,z_full) ≥ 0.10`;
2. `representation_gain = 1−[1−cos(H_gap,H_full)]/[1−cos(H_untreated,H_full)] ≥ 0.10`.

Denominators below `1e-8` mean mechanism **failed**, not infinite gain.
Report per modality/scenario and mean, plus native-unknown preservation and
number of injected positions. These are necessary, not promotion evidence.

**Non-catastrophic first-seed rule:** finite losses/predictions, no 14-GiB or
2-h breach, clean valid macro-F1 at least R1 seed1729 minus `0.05`, and clean
MAE at most R1 seed1729 plus `0.05`. No architecture/threshold adjustment
after seed1729. If full E1 fails either mechanism or this rule, do not run
remaining full seeds.

## Run order and ablations

1. Verify v2 hashes/split guard, frozen BERT/scaler/mask SHA and comparator
   records; freeze this file and model card.
2. Calibrate auxiliary weights on train only.
3. Fit **E1-full seed1729**.
4. Fit **E1-local-only seed1729**: same restoration and local loss, but
   `lambda_global=0`. Diagnostic only.
5. Fit **E1-consistency-only seed1729**: R1 content pooling, no restoration
   or local loss, but clean/gap global consistency. Diagnostic only.
6. Check first-seed stability, budget, both mechanism gates and prediction.
7. Only if full E1 is mechanism-valid and non-catastrophic, fit full E1
   seeds 2718 and 31415, unchanged.

## Frozen prediction promotion gate

Primarily against the frozen R1 run for the same seed and exact audit mask:

- Three-seed mean clean `Δmacro-F1≥−0.015`, `ΔMAE≤+0.025`.
- At least one six-scenario-mean effect: gap macro-F1 gain `≥+0.010`
  **or** gap MAE reduction `≥+0.015`.
- The corresponding paired clean-to-gap degradation improves; positive
  direction in at least 2/3 seeds and 4/6 applicable scenarios.
- Both first-seed mechanism gates remain necessary. Mechanism improvement
  alone cannot promote E1.

If three full seeds are not admissible because the first fails, report
`Q2_E1_MECHANISM_FAILED` for mechanism failure or `Q2_E1_REJECTED` for
catastrophic prediction with mechanisms valid. An execution/input failure
reports `Q2_STAGE3D_BLOCKED`. No A2 test or A3 access follows automatically.
