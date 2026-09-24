# Q2 one-time official A2 test evaluation

The final predictor was frozen in [final_pipeline_freeze.md](final_pipeline_freeze.md) before A2 test access: one M0 seed2718 checkpoint, raw-text adapter, no ensemble or calibration. This report is the one-time official **A2 unaligned test** evaluation. No model was trained, changed, reselected or tuned after seeing test results.

Test samples: **727**. Finite logits/intensity: **True**. Accuracy **0.6726**, macro-F1 **0.5987**, weighted-F1 **0.6535**; MAE **0.6274**, Pearson **0.6729**.

## Classification

Confusion matrix, true rows and predicted columns in Negative / Neutral / Positive order:

| True \ Pred | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 149 | 21 | 37 |
| Neutral | 43 | 42 | 73 |
| Positive | 38 | 26 | 298 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.6478 | 0.7198 | 0.6819 | 207 |
| Neutral | 0.4719 | 0.2658 | 0.3401 | 158 |
| Positive | 0.7304 | 0.8232 | 0.7740 | 362 |

## Output and resources

Predicted class counts: {'Negative': 230, 'Neutral': 89, 'Positive': 408}; true class counts: {'Negative': 207, 'Neutral': 158, 'Positive': 362}.
Predicted intensity range: [-2.337578535079956, 2.2618072032928467]; logit range: [-3.1247940063476562, 3.458282709121704]. BERT encoding 24.06s; M0 inference 0.65s; total 26.91s. Sampled peak process RSS 1.39 GiB.

## Evidence boundary

[A2 valid S1–S6](../q2_bridge/M0_results.md) was development/model-selection and synthetic-gap robustness evidence. This A2 test report measures one-time clean generalization only; no test perturbations were generated. Attachment 3 remains reserved for future unlabeled inference. Differences from valid performance do not change the frozen predictor.

Official test labels came from Attachment 2 `label.xlsx` (SHA-256 `0d6351ace01a2edee8861b21b2893c82d916d2a3e8b69d3b4f832ca841349875`), joined by the preserved official sample ID. Prediction table: [a2_test_predictions.csv](a2_test_predictions.csv) (SHA-256 `ed322de0e6938434c977199a76a789639ef9922fb82908f694064e7d8417f521`).
