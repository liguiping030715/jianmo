# Q2 final predictor freeze — recorded before A2 test access

The final predictor is the **single frozen M0 checkpoint from seed 2718**. The Bridge checkpoint selection rule was `0.5 × clean + 0.5 × mean(S1–S6)` A2-valid loss. Its saved values were 1.4434596101 (1729), **1.3961483439 (2718)** and 1.4760547678 (31415). Seed 2718 is the unique minimum. No multi-seed aggregation had been justified or frozen, so no ensemble is introduced for the final test. This decision uses only existing A2-valid records; it was written before any A2 test input or label was opened in this stage.

## Exact prediction rule

1. Read the official **A2 unaligned test** in its preserved row/ID order from frozen `cleaning_2026e_v2`.
2. Convert each `raw_text` using the validated frozen adapter: pinned uncased tokenizer with special tokens, right padding/truncation to 50, explicit attention and token-type rows; frozen local BERT in eval mode produces `[50,768]` float32 states. Text `P` follows attention. Do not read the A2 precomputed continuous `text` feature.
3. Pair these states with frozen standardized native audio `[500,74]`, vision `[500,35]` and their audited P/O arrays. Do not truncate vision by `vision_lengths`, impute zeros or alter test samples. M0 uses `P!=0` for key/pooling support and has no M1 state adapter.
4. Load only `M0_seed2718/checkpoint.pt` and run eval/inference mode. Polarity is `argmax` of the three raw logits (0 Negative, 1 Neutral, 2 Positive). Intensity is the model's continuous `3*tanh` output. No calibration, threshold, rounding, clipping, test augmentation, seed selection or ensemble averaging.
5. Evaluate once against the official A2 test labels, preserving each original ID and row. Report Accuracy, macro/weighted F1, confusion matrix, per-class precision/recall/F1, MAE and Pearson. A worse test result does not trigger tuning; investigate only a concrete implementation error.

## Frozen provenance

| Component | SHA-256 |
|---|---|
| M0 seed2718 checkpoint | `6b1d31d2af92a5edb84ea133350db98eb5a7a5fcdbece2debd4887b978e26c56` |
| M0 source `q2_bridge.py` | `dc0367156de2bd7108d7910557692c173c35964b5c2307beb0dd8d32a2fedccf` |
| M0 config | `781c6f60a3d7ffca440899a41b510398b9162a09f9de6aea15e01532507bc80c` |
| Text adapter source | `b8a5ba1d9a30763d1101538b7df33601dfb9f433bfdbddaa8f649092ad2a5945` |
| Tokenizer vocab / tokenizer JSON | `b49e80874c5efbc7f4cde245a812673b05ef1360ced56b8bc8c750eaeef3fe0f` / `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` |
| Tokenizer config | `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10` |
| BERT weights / config | `097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a` / `b92c83fdd39b9dcdded83e388feefd12a9bfc6e4e81cda9328c73b6865b10b3f` |
| Frozen v2 manifest / unaligned train-fitted scaler | `a1ae46325f3df842add1f5f384e16b5b0af92e487a909ee7bc67ac7d82440a83` / `7c6a50015d705c9afd7b3c98e6a729c35a92a08fda619d3be632ebacaba8bb28` |
| Cleaning contract / observation contract | `8cbc834a96275a08ab7670e6c40b1fd6060ebe1551856e9e1ee4e617aebf74c1` / `30f45ed2d440375016305ca56ef4be65d4696de4ae64f378756820b539774a41` |

The validated raw-text adapter's A2-only equivalence evidence is frozen in [the adapter audit](../q2_bridge/a3_text_adapter_audit.md). The [Bridge M0 results](../q2_bridge/M0_results.md) carry the A2-valid S1–S6 development robustness evidence. A2 test is a one-time generalization evaluation; no new test perturbation or model selection is allowed. Attachment 3 remains reserved for future unlabeled inference and is not part of this evaluation.
