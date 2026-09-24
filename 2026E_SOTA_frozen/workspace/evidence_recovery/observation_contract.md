# 2026E observation contract — Stage 2.5

**Scope and authority.** Consume only `workspace/data/processed/cleaning_2026e_v2/`; `cleaning_2026e_v1` is invalid. This contract describes an operational interface, not the official hidden missing mask. Raw files and frozen v2 arrays remain unchanged. Evidence: [cleaning report](../data_audit/cleaning_report.md), [token audit](tokenizer_compatibility.json), and the v2 arrays. No `R` ground truth is supplied.

## State definitions

| State | 0 | 1 | 2 |
|---|---|---|---|
| `P` | Known padding, based on verified attention mask or trustworthy length | Defensible position with an observed feature/token | Validity unresolved |
| `O` | Known unavailable because the position is padding | Usable observed feature/token | Availability unresolved |

For audio/vision, a nonzero feature row supports **operational** `P=O=1`: the row contains a finite recorded feature. It does not establish a physical timestamp or extractor quality. An all-zero row does not establish missingness or padding and remains `P=O=2` unless trustworthy metadata says otherwise. `zero_context=3` means an internal suspected gap in Attachment 3; it is **not** an official missing label. A genuine zero-valued feature is possible. Never derive a new state from normalized numeric values; use the frozen state arrays or the declared text tokenizer/mask rule.

## Per-version, per-modality contract

`A2` means its train/valid/test splits; `A3` and `A4` mean the special attachments. Counts are position counts, not sample counts.

| Version / modality | `P=1` known valid | `P=0` known padding | `P=2` unresolved | `O=1` usable | `O=0` unavailable | `O=2` unresolved |
|---|---|---|---|---|---|---|
| aligned / text, A2+A4 | `text_bert` attention=1, tokenizer compatible; all such positions except A3 caveat usable | attention=0 | none | attention=1 | attention=0 | none |
| aligned / text, A3 | attention=1 from supplied `text_bert[1]`, 665 positions, derived **in memory** | attention=0, 835 positions | none | 534 active token positions with ID other than 100 | 835 attention padding positions | 131 active ID-100 positions; these may represent hidden text, but no official mask exists |
| aligned / audio, all | finite nonzero rows (A2 train: 76,817; A3: 474) | none provable | zero rows (A2 train: 92,933; A3: 1,026) | same nonzero rows | none provable | same zero rows |
| aligned / vision, all | finite nonzero rows (A2 train: 72,633; A3: 470) | none provable | zero rows (A2 train: 97,117; A3: 1,030) | same nonzero rows | none provable | same zero rows |
| unaligned / text, A2+A4 | same verified tokenizer attention as aligned | attention=0 | none | attention=1 | attention=0 | none |
| unaligned / text, A3 | tokenize supplied `raw_text` with pinned tokenizer, in memory; attention=1 at 665 positions | tokenizer attention=0 at 835 positions | none | 665 generated active tokens | 835 tokenizer padding positions | none; this describes the supplied text, not a recovered hidden text mask |
| unaligned / audio, A2+A4 | nonzero prefix exactly matches `audio_lengths` in each sample | suffix after verified length | none under checked files | within verified prefix | suffix padding | none under checked files |
| unaligned / audio, A3 | finite nonzero rows (3,237) | none provable: no length supplied | zero rows (11,763) | same nonzero rows | none provable | same zero rows |
| unaligned / vision, all | finite nonzero rows (A2 train: 353,483; A3: 2,268) | none provable | zero rows (A2 train: 1,344,017; A3: 12,732) | same nonzero rows | none provable | same zero rows |

The other A2 splits and A4 follow the same rules; their exact counts remain in the frozen arrays and [cleaning statistics](../results/data_audit/cleaning_statistics.json). Unaligned `vision_lengths` is preserved only as raw metadata: 618/141/131 A2 train/valid/test samples contain nonzero rows beyond its declared length. It cannot create `P=0` or truncate content. A3 has no length field or original CMU ID. Aligned A3 `text_bert` contains exact int64-safe tokens; its active ID-100 changes occur in 27/30 paired samples compared with unaligned A3 raw text, while attention and token-type rows match. Do not infer a source phrase from those ID-100 positions or substitute cross-version text.

## Safe numerical consumption

For each position expose `(x, onehot(P=0,1,2), onehot(O=0,1,2))` with explicit integer states. Keep feature values intact, including all-zero rows. `P=0` is the only hard, evidence-backed padding exclusion. `P=2/O=2` is an unknown category, **not** a training target for missingness and not a silent synonym for `0` or `1`. An implementation using an attention mechanism must define how unknown positions enter attention, preserve their state indicators, handle an entirely unknown modality without a zero-denominator or all-masked softmax, and compare the effect of excluding unknown content as a sensitivity analysis. Such handling is an interface test, not a claim about the official missing mask. Known `P=1/O=1` remains observable; an A3 ID-100 token has `P=1/O=2` and must retain its supplied ID. Unknown features must never be imputed here.

`R` is absent from source files. Any future reliability value is model-estimated on Attachment 2 train only and must stay distinct from `P`, `O`, `zero_context`, and an official missingness label.

## Restricted aligned_50 pilot interface

**Defensible with limitations.** Text position validity is explicit; audio and vision nonzero rows are usable; zero rows remain an exposed unknown state. The fixed 50-position arrays and train-fitted scalers are consistent across A2/A3/A4. A pilot may use the tuple above, known padding exclusion, and a declared unknown-state sensitivity check. It may not claim to recover the official missing mask, train on `zero_context=3` as truth, treat native vision zeros as unavailable, or use A3/A4 to tune unknown handling. The full unaligned interface is also inspectable, but visual validity is much less determined and its declared `vision_lengths` cannot resolve it.
