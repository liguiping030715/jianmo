# Q1 final semantic review

## Decision

**Q1_COMPLETE.** The accepted media extraction, verified timebase and 50-bin
alignment were retained. This patch separates transcript content from the
ability to place individual words on that timebase.

## Corrected interface

| Concept | Stored field | Meaning in these 100 samples |
|---|---|---|
| CONTENT EXISTS | `text_content_present` | 1 for all 100 official transcripts |
| TEMPORAL LOCATION VERIFIED | `text_alignment_state=1` | 78 samples; accepted word intervals may be projected to bins |
| TEMPORAL LOCATION UNRESOLVED | `text_alignment_state=2` | 22 samples; all text-bin `P=2,O=2`, with no fabricated locations |
| VERIFIED ABSENT | `text_alignment_state=0` | Defined but unused in this batch |
| OBSERVATION AVAILABLE | `O=1` | A finite local feature exists at that bin |

For all 100 records, `text_word_feat` holds frozen BERT contextual word vectors;
`text_word_char_start`, `text_word_char_end` and `text_word_surface` link them to
the original transcript; `text_utterance_feat` is their mean. These content
features are usable at sample level even when temporal placement is unresolved.
`text_feat` remains the original time-bin feature array; no word is placed by
index or spread uniformly across the clip. Even in the 78 sample-level VERIFIED
records, bins lacking a verified word are `P=2,O=2`: a partially failed word
alignment cannot prove those bins contain no transcript content. No text bin
is marked verified absent in this batch.

The original `*_feat` arrays keep nonobserved NaN placeholders. The new
`*_model_input` arrays are finite, with zero placeholders only where `O!=1`.
Consumers must use P/O and `text_alignment_state` with these numeric inputs;
placeholder zero does not signify missing text. Audio and visual P/O retain
their accepted definitions.

## Lightweight audit

The reproducible [audit record](q1_semantic_audit.json) checked all 100 feature
NPZs, per-sample provenance JSONs, the official Attachment 1 manifest and both
CSV reports:

| Check | Result |
|---|---:|
| Official transcripts preserved | 100/100 |
| BERT word and utterance content stored, finite and source linked | 100/100 |
| Alignment state stored consistently | 100/100 |
| VERIFIED | 78/100 |
| UNRESOLVED | 22/100 |
| Unresolved text mislabeled as verified absence | 0/22 |
| Original labels preserved | 100/100 |
| Finite entries wherever O=1 | 100/100 |
| Finite separate model input arrays | 100/100 |
| Frozen extractor code/config/BERT model hashes unchanged | 10/10 |
| Audit failures | 0 |

Only the frozen BERT was rerun on official transcript strings, because the
original extractor had discarded its word vectors after temporal projection.
No raw video decoding, wav2vec alignment, openSMILE or MediaPipe run was
repeated. The optional geometric landmark normalization was not needed for
this acceptance patch and was not performed. The existing
`figures/typical_alignment.png` depicts one VERIFIED example; its text-timing
track does not assert that the 22 UNRESOLVED samples lack content.

**Later traceability audit:** [q1_traceability_review.md](q1_traceability_review.md)
documents the old figure's omitted low-confidence words, the audio tail, the
effective edited duration, 100 source maps and the remaining per-frame visual
contributor limitation. The original reviewer acceptance did not establish
that each FaceMesh frame contributing to a bin can be reconstructed from the
saved aggregate, and no such stronger claim is made here.

Reproduction: `python workspace/q1_final/q1_semantic_patch.py` followed by
`python workspace/q1_final/q1_semantic_audit.py`. The patch script is
idempotent and skips BERT when word content is already present.

Q1_COMPLETE
