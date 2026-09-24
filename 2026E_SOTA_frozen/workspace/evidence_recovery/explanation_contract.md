# 2026E Q3 explanation output contract — Stage 2.5

This specifies output and validation fields before any model is selected. It applies to every Attachment 4 sample in the chosen feature version. Prediction and explanation must use the same trained Q2/Q3 model interface and frozen v2 data. Attachment 4 is final inference only; it does not tune preprocessing, attribution baselines, thresholds, or model parameters.

## Per-sample output

| Field | Required meaning |
|---|---|
| `sample_id`, `feature_version`, `source_file` | Original ID when present; Attachment 4 local ID plus its exact PKL/MP4 paths. |
| `predicted_polarity`, `predicted_intensity` | Polarity in Negative/Neutral/Positive and continuous intensity in the official range; include the decision rule and raw score/probabilities separately. |
| `classification_modality_contributions` | Text/audio/vision signed contribution to the **declared polarity score** under a specified intervention, with score unit and reference condition. |
| `regression_modality_contributions` | Text/audio/vision signed change to predicted **intensity** under the corresponding intervention, in intensity units. |
| `dominant_modality_classification`, `dominant_modality_regression` | Selected independently under a declared ranking rule, with ties/low-evidence permitted. The two dominant modalities may differ. A single `dominant_modality` field, if an export requires it, must name which target it refers to. |
| `important_interval` | Target-specific important token/feature interval(s), including modality, half-open indices, and the `P/O` states used. |
| `source_locator`, `localization_resolution`, `mapping_uncertainty` | Structured locator per [source-localization contract](source_localization.md). Text phrase where available; audio/vision feature interval; verified media time only with a demonstrated mapping. |
| `deletion_response`, `retention_response` | Original and intervened polarity score/class and intensity, intervention definition, affected interval, and change. Retention must define what was retained and what reference replaced the rest. |

Report classification and regression evidence separately when they disagree. The polarity contribution score might be the predicted-class logit margin or calibrated probability change; it must be fixed before validation and consistently labeled. A contribution is an **intervention response of this model**, not a causal effect of a real-world modality. Do not use attention weight alone as contribution.

## Faithfulness protocol that a later pilot can instantiate

Choose an intervention that respects the observation contract: change a declared usable content interval or availability state using a train-defined reference; leave IDs, labels, raw files, and all other modality inputs unchanged. Record the intervention exactly. Directly zeroing a feature is not automatically "missing", because native zero rows occur. For each reported important interval, compare deletion and retention responses with equal-length or matched-control intervals, using Attachment 2 valid for evaluation. Recheck contribution ranking and localization stability across seeds or permitted references before making a strong claim. The official A4 set is used for final explanations only.

No exact feature-to-second link, hidden missing mask, or word-level spoken timestamp is prerequisite for this contract. When mapping is unavailable, the machine output must use `null` and a named uncertainty state, never a guessed second value.
