# M0 final architecture acceptance audit

**Architecture status: `M0_ARCHITECTURE_ACCEPTED`.** Three frozen seed1729/2718/31415 M0 checkpoints; only 728 A2 unaligned valid samples. No training, patching, A2 test, A3 or A4 access. FULL reproduced the saved clean predictions and metrics for every seed. Executed `q2_bridge.py` SHA-256 `dc0367156de2bd7108d7910557692c173c35964b5c2307beb0dd8d32a2fedccf` matched all three environment records.

## A. Executed architecture

M0 consumes text **50×768** frozen BERT hidden states, audio **500×74**, and vision **500×35**. Three separate linear input projections map 768→48, 74→48 and 35→48. Fixed, nonlearned sinusoidal native positional buffers of lengths 50, 500 and 500 (48 channels) are added **after** their projections and **before** GELU. They are registered with `persistent=False`, so checkpoint tensors do not contain learned position weights.

For M0, `P!=0` is the usable attention/pooling criterion; `P=0` is excluded, `P=2` is retained. M0 does not use `O` in attention or pooling. Text BERT itself receives the supplied attention mask. Padded text positions are computed as independent queries but are excluded from all final text pooling; no text-query output influences another text query. An all-masked A/V key set gets a neutral zero attention result rather than an all-masked softmax.

Only **two** cross-modal directions exist: text Q ← audio K,V and text Q ← vision K,V. There is no audio-query, vision-query, audio↔vision or full pairwise six-direction exchange. The executed architecture is **text-anchored MulT-style cross-modal attention**, not full pairwise MulT.

| Block | Q | K | V | First residual | Norm/FFN/second residual | Dropout |
|---|---|---|---|---|---|---|
| `audio_to_text` | projected+positioned text (50×48) | projected+positioned audio (500×48) | same audio | text query + attended audio output, both 48-d | post-LayerNorm; 48→96→48 GELU FFN; second residual then post-LayerNorm | attention weights 0.1; attended output 0.1; FFN hidden 0.1 |
| `vision_to_text` | projected+positioned text (50×48) | projected+positioned vision (500×48) | same vision | text query + attended vision output, both 48-d | same post-LayerNorm/FFN/second post-LayerNorm | same 0.1 sites |

There is no 768-d text + 74-d audio or 768-d text + 35-d vision residual. All attention and residual paths are 48-dimensional. Masked mean pooling separately summarizes base text, audio-attended text and vision-attended text into three 48-vectors. Their 144-vector concatenation enters Linear(144,48)→GELU→dropout(0.1). The classification head is Linear(48,3) logits; the regression head is Linear(48,1) followed by `3*tanh`, producing a bounded continuous intensity.

## B. Position-order sensitivity

Within each sample, only P=1/O=1 native feature rows were permuted or reversed; all masks, unresolved rows, labels and other modalities were left in place. Audio had at least two eligible rows in 728/728 valid samples; vision in 727/728. Logit and intensity changes below are absolute relative to FULL. Numerical equivalence was fixed at max ≤1e-5 for both outputs before inference.

| Seed | Condition | Max abs logit Δ | Mean abs logit Δ | Mean abs intensity Δ | macro-F1 | MAE | Pearson |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1729 | FULL | 0.000000 | 0.000000 | 0.000000 | 0.6030 | 0.6111 | 0.6353 |
| 1729 | AUDIO_TEMPORAL_PERMUTED | 0.140877 | 0.013146 | 0.011093 | 0.6053 | 0.6104 | 0.6361 |
| 1729 | VISION_TEMPORAL_PERMUTED | 0.137506 | 0.007867 | 0.007190 | 0.6042 | 0.6121 | 0.6345 |
| 1729 | AUDIO_REVERSED | 0.223126 | 0.017246 | 0.013839 | 0.6079 | 0.6118 | 0.6347 |
| 1729 | VISION_REVERSED | 0.226293 | 0.011445 | 0.010826 | 0.6058 | 0.6130 | 0.6332 |
| 2718 | FULL | 0.000000 | 0.000000 | 0.000000 | 0.5877 | 0.5788 | 0.6523 |
| 2718 | AUDIO_TEMPORAL_PERMUTED | 0.088955 | 0.010231 | 0.006268 | 0.5923 | 0.5790 | 0.6518 |
| 2718 | VISION_TEMPORAL_PERMUTED | 0.166681 | 0.010673 | 0.010127 | 0.5893 | 0.5786 | 0.6526 |
| 2718 | AUDIO_REVERSED | 0.136544 | 0.015176 | 0.009040 | 0.5939 | 0.5789 | 0.6516 |
| 2718 | VISION_REVERSED | 0.295903 | 0.016757 | 0.016210 | 0.5919 | 0.5800 | 0.6515 |
| 31415 | FULL | 0.000000 | 0.000000 | 0.000000 | 0.5529 | 0.6281 | 0.6348 |
| 31415 | AUDIO_TEMPORAL_PERMUTED | 0.115726 | 0.009144 | 0.007582 | 0.5520 | 0.6287 | 0.6344 |
| 31415 | VISION_TEMPORAL_PERMUTED | 0.095786 | 0.008246 | 0.007643 | 0.5543 | 0.6287 | 0.6352 |
| 31415 | AUDIO_REVERSED | 0.119143 | 0.011440 | 0.009645 | 0.5541 | 0.6292 | 0.6347 |
| 31415 | VISION_REVERSED | 0.232291 | 0.013074 | 0.012055 | 0.5542 | 0.6277 | 0.6352 |

Order gate: audio PASS; vision PASS. The output changes establish effective native order sensitivity, but several permutations/reversals slightly improve macro-F1; this audit does not establish that the learned ordering improves predictive quality. See CSV/JSON for max regression changes and Accuracy/weighted-F1.

## C. Cross-sample modality-use diagnostic

The same label-blind deterministic donor manifest (SHA-256 `ea337c7f9b7c19612999b74c2b44520b669bf2a420253928a6d285686530e211`) was shared across all seeds. Each receiver retained its own text/label; the donor's entire native A/V sequence and P/O masks moved together. This tests use of sample-specific information and is **not** causal importance.

| Seed | Condition | Δ macro-F1 | Δ weighted-F1 | Δ MAE | Δ Pearson | Max abs logit Δ |
|---:|---|---:|---:|---:|---:|---:|
| 1729 | FULL | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
| 1729 | AUDIO_SAMPLE_PERMUTED | -0.0068 | -0.0072 | -0.0001 | -0.0012 | 1.9567 |
| 1729 | VISION_SAMPLE_PERMUTED | -0.0157 | -0.0164 | +0.0222 | -0.0295 | 1.2714 |
| 1729 | AUDIO+VISION_SAMPLE_PERMUTED | -0.0261 | -0.0256 | +0.0259 | -0.0332 | 2.0919 |
| 2718 | FULL | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
| 2718 | AUDIO_SAMPLE_PERMUTED | +0.0033 | +0.0013 | +0.0061 | -0.0103 | 1.2115 |
| 2718 | VISION_SAMPLE_PERMUTED | -0.0017 | -0.0018 | +0.0183 | -0.0226 | 1.1266 |
| 2718 | AUDIO+VISION_SAMPLE_PERMUTED | +0.0068 | +0.0063 | +0.0240 | -0.0336 | 1.4538 |
| 31415 | FULL | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 |
| 31415 | AUDIO_SAMPLE_PERMUTED | -0.0021 | -0.0036 | -0.0010 | +0.0062 | 1.9760 |
| 31415 | VISION_SAMPLE_PERMUTED | -0.0061 | -0.0045 | +0.0144 | -0.0194 | 1.0828 |
| 31415 | AUDIO+VISION_SAMPLE_PERMUTED | -0.0144 | -0.0150 | +0.0166 | -0.0159 | 1.9613 |

| Condition | Mean Δ macro-F1 | Mean Δ weighted-F1 | Mean Δ MAE | Mean Δ Pearson |
|---|---:|---:|---:|---:|
| AUDIO_SAMPLE_PERMUTED | -0.0018 | -0.0032 | +0.0016 | -0.0018 |
| VISION_SAMPLE_PERMUTED | -0.0078 | -0.0076 | +0.0183 | -0.0238 |
| AUDIO+VISION_SAMPLE_PERMUTED | -0.0112 | -0.0114 | +0.0222 | -0.0276 |

Predictive-influence gate: audio **PASS** (2/3 seeds); vision **PASS** (3/3 seeds). Vision gives the more consistent degradation: mean Δ macro-F1 -0.0078 and mean Δ MAE +0.0183. Audio's average effect is smaller and mixed across metrics.

## D. Text interface boundary

A2 uses supplied `text_bert` → pinned frozen local BERT → 50×768 states, never the precomputed A2 `text` feature. The prior A2-valid shadow probe reproduced those states from `raw_text` exactly. The frozen unaligned A3 schema instead requires `raw_text` → pinned tokenizer → same BERT; a final inference adapter is not implemented. No A3 sample was opened here, and aligned A3 `text_bert` was not mixed with unaligned A/V. This remains a separate deployment boundary after architecture acceptance.

## E. Decision

Structural gate: PASS. Audio/vision order gate: True/True. At least one non-text sample-specific influence gate: True. **M0_ARCHITECTURE_ACCEPTED**. This is an architecture verdict on A2 valid, not a final A3 deployment or Q2 answer.

Machine evidence: `m0_architecture_audit.json`, `m0_order_sensitivity.csv`, `m0_modality_ablation.csv`, and `m0_permutation_manifest.jsonl`.
