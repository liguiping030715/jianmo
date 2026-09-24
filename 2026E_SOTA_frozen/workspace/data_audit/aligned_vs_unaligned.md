# Aligned versus unaligned — empirical audit, no winner selected

| Evidence | aligned_50 | unaligned_50 |
|---|---|---|
| Attachment 2 size | 993,842,861 bytes | 2,897,003,035 bytes (2.92×) |
| Text | `(N,50,768)`; attention-mask valid mean 24.65 train | Same text and valid mask |
| Audio | `(N,50,74)`; first vector and suffix zero; mean last nonzero position 23.65 train | `(N,500,74)`; mean valid length 147.29 train; length field exact, right padding |
| Vision | `(N,50,35)`; 110 train all-zero samples; 210 train interior zero runs | `(N,500,35)`; mean stated length 93.25 train; 618 train length fields shorter than observed support |
| Local-gap resolution | Coarse shared positions, but audio/vision zero patterns and text attention positions differ | Finer native positions, but cross-stream correspondence and visual length metadata are unreliable |
| Explanation localization | Common index convenient; index-to-video seconds undocumented | More audio/video positions; index-to-video seconds still undocumented |
| Compute | Lower sequence/storage burden | ~10× audio/vision positions, ~2.92× on-disk PKL; actual model cost depends on architecture |
| Attachment 3 interface | Only `text_bert` (float32), audio, vision; no precomputed `text` or ID | Only `raw_text`, audio, vision; no precomputed `text`, text tokens, lengths or ID |
| Attachment 4 interface | Full text, audio, vision and local ID | Full features, lengths and local ID; vision-length caveat remains |

The versions preserve different time detail; aligned is not automatically better. Neither PKL supplies word-level or frame-level timestamps. A common position is not proof of exact second-level correspondence. Missing-interval size in position units is not comparable between 50 and 500 without a validated time map. Train/valid feature scales are similar within each modality (audio observed-value mean ~2.30–2.35, SD ~21–22; vision mean ~-0.86 to -0.91, SD ~2.5–2.6), but this does not establish performance equivalence.

**aligned_50: SUPPORTED_FOR_PILOT** as an Attachment 2 representation, provided the pilot uses the same `text_bert` pathway for its Attachment 3 interface and does not assert exact missing-mask ground truth. **unaligned_50: SUPPORTED_FOR_PILOT** as an Attachment 2 representation only with an explicit rule for the non-prefix `vision_lengths` behavior and a consistent `raw_text` encoder for Attachment 3. These labels permit *interface pilots*, not final Q2 inference readiness. The critical mask gate keeps overall Stage 2 `BLOCKED`. Neither version is selected.
