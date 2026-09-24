# Q2 bridge pilot — frozen before M0/M1 training

Decision scope: H11 tests whether a small native-index MulT-style backbone is a viable Q2 predictor; H12 isolates an audited observation-state adapter. E0_REJECTED and E1_MECHANISM_FAILED remain frozen. RU (unaligned lightweight) and R1 (aligned lightweight) are read-only comparators. RU versus M0 is a backbone/interface comparison, never an adapter effect.

## Data and interface

- Only `workspace/data/processed/cleaning_2026e_v2/unaligned/{train,valid}`: 3,395/728 rows. No Attachment 2 test, Attachment 1, Attachment 3 or Attachment 4 read for Q2 training, tuning, or prediction.
- Native dimensions: text 50×768 frozen local BERT states from the same `text_bert` IDs, audio 500×74, vision 500×35. Text gap tokens use the frozen RU BERT gap cache. No force alignment or vision-length truncation.
- Reuse RU's seed-specific S1–S6 native-index JSONL masks byte for byte. Each sample contributes one clean and one selected gap copy to training. Validation selection uses the frozen `selection` masks; final audit uses `audit` masks.
- An injected gap replaces an eligible text ID with `[UNK]` before frozen BERT or zeroes eligible audio/vision feature rows and sets their `O=0`. These are declared synthetic interventions, not recovered official masks. Native index bands do not assert physical cross-modal alignment.

## Frozen architecture

Both M0 and M1 use identical projections, sinusoidal native positional encodings, two directional cross-modal attention blocks (`text queries → audio keys/values`, `text queries → vision keys/values`), one residual feed-forward block after each attention, masked mean text pooling, 48-wide GELU fusion, a 3-class head, and a tanh-bounded intensity head. Projection width 48, 2 attention heads, 1 cross-attention layer per direction, feed-forward width 96, dropout 0.1. There is no self-attention, restoration, distillation, consistency, reliability gate, MoE, or hidden missing-mask reconstruction. The two cross-attention directions keep the source modalities on native 500-step grids; no common 50-step A/V resampling is done.

- M0: plain projected features plus native position. Verified `P=0` positions are excluded from attention/pooling. `P=2` stays available. No P/O embeddings and no O-dependent attention mask; thus the synthetic zeroed gap remains a numeric input.
- M1: identical backbone plus per-modality learned `P` and `O` embeddings (3 states each) added after projection. `P=0` and defensible `O=0` are excluded as keys or pooling positions. `P=2/O=2` remain available with explicit unknown embeddings. A fully masked key set gets a neutral zero attention output, avoiding NaN. Native zeros do not create new missing labels.
- M0/M1 use the same hidden size, heads, depth, fusion, heads, loss, optimizer, schedule, batches, masks and seeds. The adapter is the only model/interface difference. No redesign after seed 1729.

## Frozen fitting and selection

- Seeds 1729, 2718, 31415; deterministic PyTorch CPU, 8 threads. Batch 32 for fitting/inference. AdamW learning rate 0.001, weight decay 0.0001, no LR scheduler. Maximum 12 epochs, patience 3, minimum selection improvement 1e-6.
- Loss `cross_entropy + 2.0526115894317627 × Huber(delta=1.0)`; intensity output `3 tanh(raw)`. No class reweighting or label change.
- Checkpoint by `0.5 × clean selection loss + 0.5 × mean(S1–S6 selection losses)` on A2 valid. This is the previously used RU selection form. Audit metrics are calculated only after checkpoint selection.
- Resource gate: peak process-tree RSS ≤14 GiB and projected wall clock ≤7,200 seconds per seed. If the first M0 epoch projects beyond the cap, stop with `Q2_BRIDGE_BLOCKED`; do not simplify the architecture. Nonfinite loss, missing/changed source checksums, or a failed run is recorded and never silently discarded.

## Order and technical gate

1. Preflight v2 manifest, scaler and BERT hashes, train/valid shapes/counts and P/O domain; verify all three frozen masks exactly by RU's generator. Record environment and source checksums.
2. M0 seed 1729. Continue only if finite, complete, within resource cap, and prediction files contain all 728 valid IDs for clean and six audit conditions.
3. M1 seed 1729, with the identical seed-1729 mask bytes. Continue only if it meets the same technical gate.
4. If both first-seed runs are technically valid, run M0 then M1 for 2718 and M0 then M1 for 31415. No result-based architecture or threshold edits.

## Frozen decision rules

Report Accuracy, macro-F1, weighted-F1, MAE, Pearson on clean and each injectable paired gap set, clean→gap changes, seed/scenario variance and resources. An uninjectable sample remains in clean evaluation and is excluded only from that scenario's paired gap comparison.

H11 support requires all three M0 seeds technically valid; M0 mean clean macro-F1 ≥ frozen RU mean −0.015 and mean clean MAE ≤ RU mean +0.025; at least 2/3 seeds satisfy both paired clean bounds; mean six-scenario gap macro-F1 ≥ RU mean gap macro-F1 −0.015 and mean gap MAE ≤ RU mean gap MAE +0.025; and the resource cap holds. This is a competitive viability gate, not evidence of MulT superiority. R1 is contextual only because its interface is aligned.

H12 promotion requires all three M0/M1 seed pairs technically valid; mean M1−M0 clean macro-F1 ≥−0.015 and clean MAE increase ≤+0.025; across six scenarios the mean paired M1−M0 gap macro-F1 ≥+0.010 **or** mean paired M0−M1 gap MAE ≥+0.015; the chosen positive metric has positive direction in at least two of three seed aggregates and at least four of six scenario aggregates. Use the same mask/eligible intersection within every pair. Thresholds cannot be revised after results.

The final label is `MULT_OBSERVATION_ADAPTER_SUPPORTED` if H11 and H12 pass, else `MULT_BACKBONE_SUPPORTED` if H11 passes, else `MULT_BACKBONE_NOT_SUPPORTED` after valid runs. A technical/resource failure gives `Q2_BRIDGE_BLOCKED`. No A3 predictions, restoration, or full experiments follow automatically.
