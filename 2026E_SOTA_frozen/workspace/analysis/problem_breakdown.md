# Stage 1 problem breakdown — 2026E

Source: official statement in `workspace/problem/`. No PKL, video, or XLSX contents were inspected at this stage.

| Task | Official input | Required output | Evaluation and handoff |
|---|---|---|---|
| Q1 | Attachment 1: 100 English clips in 37 video-ID folders, `label-100.xlsx` with transcript, intensity and polarity | Three defined temporal feature streams and explicit alignment rule; complete 100-sample feature files; per-sample ID, duration, dimensions, alignment granularity, valid length, padding and source mapping; tool versions and logs; at least one manually checked clip | Coverage, one-to-one file/ID mapping, temporal traceability and reproducibility. Provides the representation and provenance discipline used in Q2–Q3, but Q2–Q3 train on Attachment 2. |
| Q2 | Attachment 2 train/valid (one consistent aligned or unaligned version); Attachment 3 unlabeled features with local contiguous zero intervals | Polarity class and continuous intensity for every Attachment 3 sample; robust model, training objective and parameters; validation metrics and degradation by missing modality, position, duration and ratio | Accuracy/F1 and MAE/Pearson on valid. Tune only on train/valid; Attachment 3 final inference only. Passes mask/reliability-aware predictive representation to Q3. |
| Q3 | Attachment 2 train/valid; Attachment 4 unlabeled complete features and videos | Polarity/intensity plus per-sample modality contribution, dominant modality, local evidence linked to phrase/audio interval/frame; all Attachment 4 predictions and explanations | Same predictive metrics on valid, plus traceability and perturbation faithfulness. Attachment 4 final inference only. |

Official polarity: label < 0 Negative; exactly 0 Neutral; > 0 Positive; intensity in [-3,3]. Preserve Neutral as its own class. Attachment 2 has 4,850 samples per version and train/valid/test top-level splits. `text` is precomputed 50×768; `text_bert` is an alternative text input, not a fourth modality. Audio is 74-dimensional and vision 35-dimensional. Aligned streams have at most 50 corresponding positions; unaligned audio/vision have at most 500 positions with length fields. Feature PKLs lack raw frames, waveforms and word timestamps, so index-to-media claims require separate evidence.

Cross-question bottlenecks: sample key `(video_id,clip_id)`/`id`, time or sequence coordinate, valid/padding/missing masks, modality availability/reliability, shared predictive state, and source-linked evidence intervals. Exact index-to-seconds map and zero/padding semantics remain unresolved until Stage 2.

Constraints: only official CMU-MOSEI series data for training, fine-tuning, optimization, thresholds and result statistics; open-source pretrained feature tools allowed with version disclosure; fixed feature version across training/validation/special tests; raw Attachment 1 samples and labels retained; submission ≤50 MB and anonymous. No model selected in Stage 1.
