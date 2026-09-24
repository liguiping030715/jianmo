# Attachment 2 train/valid label audit

| Split | N | Negative `0.0` | Neutral `1.0` | Positive `2.0` | Intensity range | Unique intensities |
|---|---:|---:|---:|---:|---|---:|
| train | 3,395 | 967 | 758 | 1,670 | [-3,3] | 23 |
| valid | 728 | 206 | 184 | 338 | [-3,3] | 22 |

Direct cross-tab: all 967/206 negative classes have regression label <0; all 758/184 neutral classes equal exactly 0; all 1,670/338 positive classes have label >0. No contradiction, NaN, Inf or out-of-range intensity. Neutral is substantial (22.3% train, 25.3% valid) and remains a separate class. Within each split PKL IDs are unique; Attachment 2 spreadsheet IDs/labels/modes match the PKL exactly. Repeated intensity values are expected discrete annotations, not duplicate samples. No labels from Attachment 3/4 were used.

The spreadsheet for Attachment 1 independently has 18 Negative, 25 Neutral, 57 Positive rows and agrees with its intensity signs; it is not Q2/Q3 training evidence. Machine-readable counts: `workspace/results/data_audit/label_statistics.json` and `xlsx_rows.json`.
