# E1 mechanism diagnostics

Gate definitions were frozen in `q2_stage3d_plan.md` before valid predictions. Local gain uses Huber(z_hat,z_full) against Huber(0,z_full); representation gain uses the same E1 weights with versus without injected-position restoration. Both require ≥10% in the equal-scenario mean.

| Scenario | Injected latent positions | Local restoration gain | Fused representation gain | Native unknown positions left unrestored |
|---|---:|---:|---:|---:|
| S1 | 1754 | +70.787% | -12.126% | 39482 |
| S2 | 4282 | +77.600% | +40.147% | 39090 |
| S3 | 5132 | +73.120% | +40.206% | 37110 |
| S4 | 8560 | +68.780% | -16.699% | 38896 |
| S5 | 10661 | +72.252% | -5.067% | 37351 |
| S6 | 3349 | +81.956% | +51.307% | 38203 |

Six-scenario mean latent Huber: restored **0.077661**, zero reference **0.298650**; restoration gain **+74.00%** (passes).
Mean cosine distance: restored **0.006403**, untreated gap **0.005862**; representation gain **-9.23%** (fails).

The local target was detached clean latent state at explicit injected positions only. The E1 pilot did not restore native O=2 positions or use zero rows/A3 as target truth.
Train-only calibration: mean prediction/local/global losses 3.177917/0.329100/0.003007; lambda_local=0.965640, lambda_global=10.000000. The latter hit the prespecified cap of 10; no valid-driven adjustment was made.

Mechanism gate: **FAILED**. A large latent fit against zero did not translate into closer fused clean/gap representations.
