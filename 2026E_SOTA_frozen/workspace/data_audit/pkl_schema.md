# Direct PKL schema audit

Complete split-by-field Python type, dtype, shape, finite min/max, NaN/Inf and zero counts: [pkl_schema.json](pkl_schema.json). The JSON is generated from direct unpickling, not the statement. NumPy-2 pickle module aliases were applied for Attachment 4 compatibility with the installed NumPy 1.24.

| Version | Split N | Text | Audio | Vision | Other keys |
|---|---:|---|---|---|---|
| aligned | 3395 | [50, 768] float32 | [50, 74] float64 | [50, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels |
| aligned | 728 | [50, 768] float32 | [50, 74] float64 | [50, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels |
| aligned | 727 | [50, 768] float32 | [50, 74] float64 | [50, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels |
| unaligned | 3395 | [50, 768] float32 | [500, 74] float64 | [500, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels, audio_lengths, vision_lengths |
| unaligned | 728 | [50, 768] float32 | [500, 74] float64 | [500, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels, audio_lengths, vision_lengths |
| unaligned | 727 | [50, 768] float32 | [500, 74] float64 | [500, 35] float64 | raw_text, id, text_bert, classification_labels, regression_labels, audio_lengths, vision_lengths |

`annotations` is absent in both PKLs. `classification_labels` and `regression_labels` are float64 vectors `(N,)`; `text_bert` is int64 `(N,3,50)` and `raw_text` is a string array. `id` is a list of `video_id$_$clip_id`. Only unaligned has `audio_lengths` and `vision_lengths`, both Python lists. The statement’s principal feature shapes are verified. All numeric fields in Attachment 2 have zero NaN and zero Inf.

Attachment 3 differs from training: aligned files contain only `text_bert` (float32 `(1,3,50)`), audio and vision; unaligned files contain only `raw_text`, audio and vision. They have no `id`, `text`, labels or length fields. Attachment 4 has `id`, `raw_text`, `text`, `text_bert`, audio and vision; unaligned also has scalar length fields. See `workspace/results/data_audit/special_schema.json` for all 100 special PKLs.