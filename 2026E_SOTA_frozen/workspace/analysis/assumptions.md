# Assumptions register

| ID | Status | Statement / consequence | Evidence needed |
|---|---|---|---|
| A1 | Official | Attachment 1 contains 100 samples; no sample or label may be removed or replaced. | Coverage audit later. |
| A2 | Official | Q2/Q3 parameters use Attachment 2 train; model/threshold selection uses valid; special sets are unlabeled final inference. | Enforce split-aware pipeline. |
| A3 | Official | Local continuous zero intervals indicate designed missingness in Attachment 3. | Need distinguish from padding and legitimate zero values empirically. |
| A4 | Working, unverified | A common interval coordinate can connect streams and evidence. | PKL lengths, video duration and alignment metadata. |
| A5 | Working, unverified | Missingness masks can be recovered without target labels. | Compare train/valid/Attachment 3 fields and zero runs. |
| A6 | Working, unverified | Attachment 4 feature indices can be localized reliably in supplied videos. | Manual source checks; timestamps may require reconstructed alignment and uncertainty bounds. |
| A7 | Explicitly not assumed | Aligned is better; zero is neutral; attention is explanation; text always dominates; pretrained representation is extra modality. | Pilot or rejection audit. |

All model-derived explanations are evidence about a model decision. They are not claims about the speaker's psychological cause without separate labels and validation.
