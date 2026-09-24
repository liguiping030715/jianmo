# E1 — latent local restoration with clean/gap consistency

**Status:** preregistered Stage 3D candidate. Detailed frozen protocol:
`workspace/experiments/q2_stage3d_plan.md`.

E0's local weighting responded to injected gaps but failed stable predictive
improvement. E1 tests a distinct mechanism: estimate a missing **128-D latent
position** from observed same-modality neighbors and synchronous observed
modalities, then penalize discrepancy from that position's clean-view latent
state and from the complete-view fused representation. This is a pilot
hypothesis, not evidence that the hidden A3 mask is known or that any
raw feature is recoverable.

| Item | Specification |
|---|---|
| Input | Frozen aligned_50 text BERT states, scaled audio/vision, tri-state P/O, explicit train/valid synthetic gap mask |
| Output | Polarity logits and bounded intensity, plus diagnostic restored latent states and fused H |
| Simple comparator | Frozen R1 B1 shared fusion with mean content pooling; R0a augmentation reference |
| Sole addition | Three small per-modality restoration MLPs and two training losses; R1 fusion/heads unchanged |
| Source of restoration truth | Clean-view latent state of the same train sample, detached, only at known injected positions |
| Unknown policy | Native O=2 retained under R1; never used as reconstruction target or treated as known gap |
| Compute | CPU, same 12-epoch/3-patience optimizer protocol; extra paired clean forward on gap batches |
| External data | None; existing frozen pretrained BERT only |
| Downstream Q3 | Intervention-based explanation can target selected prediction model, but restored latent vectors are not source-level explanations |

The current problem supports synthetic contiguous-gap tests and paired
complete/partial train views. It does **not** identify real special-test
missing intervals or supply ground-truth latent states. The model therefore
restores only known injected positions in this pilot. Literature on masked
multimodal learning motivates the component, but does not authorize a
prediction claim for A3 or source-level causal interpretation. Historical
patterns inform controlled baseline comparisons, not equations or scores.

**Falsification:** <10% latent restoration gain, <10% fused-representation
distance reduction, unstable/invalid training, clean guard failure, or no
frozen six-gap predictive benefit rejects the mechanism or candidate under
the exact gate in the plan. Diagnostic ablations isolate each loss; they
cannot override full-model promotion rules.
