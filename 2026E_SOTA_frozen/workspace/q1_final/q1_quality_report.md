# Q1 Quality Report

- samples processed: 100
- source files present: 100/100
- finite observed features and finite `*_model_input`: 100/100
- P/O consistent (O=1 => P=1): 100/100
- official transcript and BERT word/utterance content preserved: 100/100
- text alignment state present: 100/100; VERIFIED (1): 78/100; UNRESOLVED (2): 22/100
- UNRESOLVED means temporal location unknown; it does **not** mean text absent.

## Verified duration vs old manifest
- old duration overestimates (rel): mean 0.293

## Verified bin coverage (median of O=1 bins / 50)
- text: median 35, mean 28.8; 22 samples have O=2 in all text bins because their temporal location is unresolved. In the other 78 samples, any bin without a verified aligned word is also O=2, because failed words may occur there.
- audio: median 50, mean 49.7, zero-coverage samples 0
- vision: median 50, mean 40.9, zero-coverage samples 10

The 22 all-unknown text-bin records retain official transcript, word-level BERT
embeddings and an utterance-level BERT summary. No word timestamps were inferred.
Unobserved entries in the original `*_feat` arrays may be NaN; the separate
`*_model_input` arrays are finite zero placeholders governed by P/O state.

The source-mapping audit found 100/100 feature bundles, 100/100 provenance records,
100/100 maps and 5,000 mapped common bins. Per-bin video frame IDs/PTS are
exact candidate frames, while exact per-frame FaceMesh contributors cannot be
identified from the originally saved aggregate features. See
`q1_traceability_review.md` and `q1_traceability_audit.json`.

## Anomalies
- none

## Text-UNRESOLVED samples
- -9y-fZ3swSY$_$4
- -HwX2H8Z4hY$_$2
- -NFrJFQijFE$_$1
- -NFrJFQijFE$_$2
- -UuX1xuaiiE$_$0
- -hnBHBN8p5A$_$6
- -hnBHBN8p5A$_$7
- -iRBcNs9oI8$_$3
- -iRBcNs9oI8$_$6
- -iRBcNs9oI8$_$7
- -iRBcNs9oI8$_$8
- -iRBcNs9oI8$_$9
- -mJ2ud6oKI8$_$1
- -mJ2ud6oKI8$_$2
- -mJ2ud6oKI8$_$8
- -mJ2ud6oKI8$_$9
- -ri04Z7vwnc$_$0
- -ri04Z7vwnc$_$2
- -ri04Z7vwnc$_$5
- -yRb-Jum7EQ$_$1
- -yRb-Jum7EQ$_$5
- -yRb-Jum7EQ$_$6
