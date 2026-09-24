# 2026E Stage 4 first-seed screen (seed 1729)

**Execution boundary:** frozen `cleaning_2026e_v2/aligned` train/valid only. [Smoke check](smoke_check.json) passed; no A2 test or Attachment 3/4 data were read for fitting/selection. Six scenarios are **model screening only**: modality, position, duration and ratio are coupled and independent factor effects cannot be inferred. The exact aligned mask manifest SHA-256 is `26b6f79e2cf83a90bd0962ea77d43ff186b03cd7d871abb6618e3810f8cd95bb` in every first-seed run.

R0 clean BERT cache took **86.97 s**; its mean training epoch took **1.37 s**, peak RSS **1.10 GiB**, best epoch 5. The per-seed 2-hour/14-GiB budget was not approached. The selected common regression-loss weight is in [lambda_selection.json](lambda_selection.json). No architecture, threshold or gap scenario was revised after seeing results.

| Run | Clean Accuracy | Clean macro-F1 | Clean MAE | Clean Pearson | Mean six-gap macro-F1 | Mean six-gap MAE | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 B0 clean | 0.635 | 0.599 | 0.602 | 0.642 | 0.582 | 0.632 | 5 |
| R0a B0 same gap augmentation | 0.625 | 0.586 | 0.613 | 0.620 | 0.564 | 0.627 | 3 |
| R1 B1 tri-state | 0.636 | 0.615 | 0.604 | 0.635 | 0.594 | 0.619 | 3 |
| R1x same R1 weights, unknown excluded at inference | 0.611 | 0.560 | 0.638 | 0.610 | 0.559 | 0.648 | inference only |
| R2 C1 dynamic gate | 0.600 | 0.514 | 0.643 | 0.638 | 0.504 | 0.647 | 2 |
| C2 clean teacher | 0.610 | 0.545 | 0.595 | 0.639 | not tested | not tested | 6 |

Gap means weight S1–S6 equally and use each scenario's same eligible samples for its clean-to-gap delta. Eligibility counts were 728/724/700/722/703/713; no sample was removed from frozen data. Full per-scenario Accuracy, F1, MAE, Pearson, paired deltas, predictions and failures are in each [run directory](../../experiments/pilot_runs/).

**First critical comparison:** R1 exceeds R0a's mean gap macro-F1 by **0.0307** but lowers mean gap MAE by only **0.0081**, below the frozen **0.025** MAE threshold. Its F1 degradation does not shrink versus R0a (mean delta −0.0212 vs −0.0203), and its MAE degradation does not shrink (mean +0.0122 vs +0.0103). Thus first-seed B1 evidence does **not** pass promotion. Additional seeds are warranted to assess stability because the F1 signal is positive; no threshold will be relaxed. R1x's fixed-weight unknown exclusion reverses R1's advantage over R0a, a material sensitivity warning.

**C1:** the gate did not numerically collapse (`max α>0.95` on 0% of clean samples; mean entropy 0.999 nats). Full-modality masking changed the targeted α by −0.0227 text, −0.0196 audio and −0.0026 vision on average; [gate intervention record](../../experiments/pilot_runs/R2_seed1729/gate_interventions.json) is inference-only and **not** an explanation. Despite responsive gates, R2 clean macro-F1 is **0.101** below R1 and gap macro-F1 **0.091** below R1; the first seed breaches the frozen clean-performance guard. It remains recorded as negative evidence. Later seeds, if run for completeness, cannot rescue it by redesign.

**C2 teacher:** its first clean macro-F1 **0.545** is **0.054** below R0's **0.599**, failing the frozen 0.01 first-seed comparison, though MAE is slightly lower. The teacher gate requires competence in at least two seeds; **no student is trained yet**. Run the remaining teacher checks before deciding `C2_TEACHER_REJECTED`.

R0a/R1, R1x, C1 and teacher run records each contain configuration, identical first-seed mask manifest, training log (empty but present for inference-only R1x), metrics, valid predictions, failure log (empty on success), and environment/checksum record. Results remain pilot evidence, not full-experiment or special-test claims.
