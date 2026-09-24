# Q2 Attachment 3 final inference

The permanently frozen **M0 seed2718** model ran once on the 30 official Attachment 3 unaligned files in order 01–30. The external file/order key is preserved; no internal sample ID was required. No training, tuning, threshold change, ensemble, imputation, smoothing, or sample removal occurred.

## Frozen input interface

`raw_text` → validated pinned tokenizer → frozen BERT → 50×768; native unaligned audio 500×74 and vision 500×35 came from frozen `cleaning_2026e_v2`. Aligned Attachment 3 text was not used. Text attention mask became text P/O. Audio/vision P/O were read unchanged from the frozen arrays; M0 excludes P=0 only and does not infer missingness from feature magnitude.

All 30 texts were available; BERT produced 665 valid and 835 padded token positions. Inputs and predictions were finite. Predicted class counts: {'Negative': 6, 'Neutral': 5, 'Positive': 19}. Intensity range: [-1.3342658281326294, 1.7526023387908936].

## Observable zero intervals, separate from prediction

Counts below describe exact all-zero feature rows in the frozen input. These are **not certified missing labels**. Interior runs are candidates; prefix, suffix, and whole-sequence runs have additional padding/content ambiguity. `zero_context=3` is diagnostic only. Duration is in native index steps and relative fraction of 500; no verified index-to-seconds map is asserted.

| Modality | Zero rows / 15,000 | Ratio | Runs | Interior | Prefix | Suffix | Whole | Median run steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audio | 11763 | 0.7842 | 635 | 602 | 3 | 30 | 0 | 1.0 |
| vision | 12732 | 0.8488 | 460 | 424 | 6 | 30 | 0 | 1.0 |

## Artifacts and provenance

- [a3_final_predictions.csv](a3_final_predictions.csv): official file order, class index/polarity, intensity, raw logits and derived probabilities.
- [a3_missingness_descriptive.csv](a3_missingness_descriptive.csv): every observable zero interval with native and relative positions; no fabricated missing mask.
- [a3_final_inference.json](a3_final_inference.json): input checks, frozen hashes, adapter settings, runtime and full descriptive summary.

BERT encoding 0.77s; M0 inference 0.06s; total 2.74s; sampled peak RSS 1.12 GiB.

Attachment 3 has no evaluation labels in this path. These outputs are predictions and descriptive input statistics, not measured accuracy or certified missingness.
