# Q3 Stage 1 explanation-method review

**Decision: Q3_EXPLANATION_METHOD_ACCEPTED.** Frozen M0 seed2718, A2 valid 728 only. No model update or special/test access. The operator and acceptance rule were written in [q3_explanation_method.md](q3_explanation_method.md) before scoring.

## Modality donor responses

The [per-sample table](q3_modality_contribution.csv) contains the original predicted-class logit response and absolute regression response to deterministic cross-sample donor replacement. These are model intervention responses, not causal effects. Classification and regression scores and dominant modalities are separate.

## Local evidence and faithfulness

Local [text](q3_local_text_evidence.csv), [audio](q3_local_audio_evidence.csv), and [vision](q3_local_vision_evidence.csv) tables give every evaluated candidate interval. Audio/vision use native indices only; seconds are unavailable. Text phrases are tokenizer reconstructions, not verified spoken timestamps.

| Modality | Target | Eligible | Top mean | Random mean | Bottom mean | Top–random 95% CI | Random–bottom 95% CI | Pass |
|---|---|---:|---:|---:|---:|---|---|---|
| text | cls | 644 | 0.808922 | 0.420289 | 0.010279 | [0.359286, 0.418506] | [0.380206, 0.442953] | True |
| text | reg | 641 | 0.461985 | 0.300571 | 0.171816 | [0.140643, 0.182028] | [0.111383, 0.148036] | True |
| audio | cls | 728 | 0.045614 | 0.001404 | -0.043970 | [0.041072, 0.047579] | [0.042280, 0.049055] | True |
| audio | reg | 728 | 0.014630 | 0.008581 | 0.001862 | [0.003919, 0.009051] | [0.005851, 0.007722] | True |
| vision | cls | 726 | 0.032261 | 0.004250 | -0.013356 | [0.024956, 0.031112] | [0.015370, 0.020190] | True |
| vision | reg | 726 | 0.026816 | 0.010023 | 0.001441 | [0.014450, 0.019336] | [0.007199, 0.010159] | True |

The top/random/bottom test uses joint equal-count deletions after ranking individual intervals on the same valid rows. It checks internal consistency of the declared operator; in-sample ranking and reference choice limit external interpretation. Retention responses and skip counts are in [q3_faithfulness.json](q3_faithfulness.json).

## Stability

| Modality | Target | Prediction-preserving rows | Top-1 agreement | Mean top-3 Jaccard |
|---|---|---:|---:|---:|
| text | cls | 56 | 0.6428571428571429 | 0.6267857142857143 |
| text | reg | 115 | 0.5652173913043478 | 0.5486956521739131 |
| audio | cls | 128 | 0.9609375 | 0.96484375 |
| audio | reg | 127 | 0.9921259842519685 | 0.9763779527559056 |
| vision | cls | 128 | 0.9765625 | 0.9921875 |
| vision | reg | 128 | 0.9921875 | 1.0 |

The stability perturbation is a small declared masking or `[UNK]` intervention. Only unchanged-class, ≤0.1-intensity-change cases enter its overlap estimate; excluded counts remain in JSON.

## Provenance and limits

Checkpoint SHA-256 `6b1d31d2af92a5edb84ea133350db98eb5a7a5fcdbece2debd4887b978e26c56`; donor manifest SHA-256 `7460ec88dfb492054c1df797922c657c7ed680b5b37a14e5a174f983a428ee3c`. Runtime 266.7 s. No attention-weight attribution, certified hidden missing mask, physical-second localization, or causal claim is made.
