# Q2 Stage 3D review packet

## Frozen boundary

E0 remains `E0_REJECTED`. E1 used only frozen A2 aligned_50 train/valid, BERT, scalers, P/O and byte-identical gap manifests. Frozen R0a/R1/E0 metrics and configs matched their preflight SHA-256 values. No A2 test, Attachment 1/3/4, retraining of comparators, or architecture change after seed1729.

## Execution

The plan was frozen before calibration and valid prediction. Calibration used the first 256 injectable train entries only. E1-full seed1729, E1-local-only seed1729 and E1-consistency-only seed1729 completed; all successes and logs remain in `stage3d_runs/`.

E1-full best epoch 2, train wall 66.4s, peak RAM 1.63 GiB; no budget breach.

## Frozen gates

- Local restoration gain +74.00% versus required +10%: **pass**.
- Clean/gap representation-distance gain -9.23% versus required +10%: **fail**.
- First-seed clean macro-F1 Δ versus R1 -0.1246 versus non-catastrophic bound −0.0500 and final guard −0.0150: **fail**.
- Clean MAE Δ +0.0187 versus guard +0.0250: **pass**.
- Six-gap macro-F1 gain -0.1224; MAE reduction -0.0291: **neither predictive effect passes**.

The mechanism gate and first-seed non-catastrophic rule both fail. Per the preregistered stop, E1-full seeds 2718 and 31415 were **not run**. Three-seed promotion cannot be assessed and E1 cannot be promoted. No parameters, thresholds or masks were changed to rescue the candidate.

Details: [comparison](E1_comparison.md), [ablations](E1_ablation_seed1729.md), [mechanism](mechanism_diagnostics.md), [machine decision](stage3d_decision.json).

Q2_E1_MECHANISM_FAILED
