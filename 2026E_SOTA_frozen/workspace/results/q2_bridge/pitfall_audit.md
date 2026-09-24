# Q2 Bridge read-only interface audit

**Verdict: `BRIDGE_TEXT_INTERFACE_INVALID` under the requested strict A3 field requirement.** The A2 training text itself is sourced correctly: M0/M1 never read the precomputed A2 `text` feature. They use the frozen `text_bert` triplet, the pinned local BERT and cached 50×768 hidden states. The gap cache re-encodes intervened token IDs with the same BERT. Code path: `pilot_ru.load_split` → `pilot_stage4.cache_clean_bert` / `pilot_ru.cache_gap_bert_ru` → `pilot_ru.build_input` → `q2_bridge.NativeMulT.forward`. The pinned BERT weights have SHA-256 `097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a`. The cache contents were checked against that encoder in bridge preflight. The older RU gap metadata prints a stale weight-hash literal; cache content matched a fresh forward pass.

The strict requested final interface is the unresolved issue. M0/M1 are **unaligned_500** models. The frozen [text interface contract](../../evidence_recovery/text_interface.md) says **unaligned A3 supplies `raw_text`**, which can be tokenized with the same pinned tokenizer into an in-memory `text_bert` triplet; **aligned A3 supplies `text_bert`**. The current bridge has no A3 inference runner, and there is no supplied same-version unaligned A3 `text_bert` field to read directly. Taking aligned A3 tokens together with unaligned audio/vision would mix version-specific content semantics. The A2 train/valid raw text was tokenized read-only with the pinned tokenizer: all **3,395/3,395** train and **728/728** valid triplets exactly matched the stored A2 `text_bert`. This supports the *derived* unaligned path, but it does not establish the requested literal supplied-A3-`text_bert` path. No A3 sample was opened or predicted in this audit. No model was patched or rerun.

## Padding and attention

| A2 split | Valid text tokens | Padded text tokens | `text_P == attention_mask` | `text_O == attention_mask` on clean A2 |
|---|---:|---:|---|---|
| train (3,395) | 83,672 | 86,078 | yes | yes |
| valid (728) | 18,628 | 17,772 | yes | yes |

BERT receives `text_bert[1]` as `attention_mask`. M0 pools text with `P!=0`; M1 pools with `P!=0` and `O!=0`. There is **no unmasked mean over all 50 positions**. Padded text positions are nevertheless computed as cross-attention **queries** to audio and vision; the bridge does not skip them at the query side. Each query is independent, text is not used as a cross-attention key/value, and the padded query outputs are excluded from every pooled summary, so they cannot alter the prediction through this architecture. A stricter rule that no padded position may enter any attention computation is **not met** and would require a separately reviewed implementation change. This finding is not silently patched.

## Audio position 0

For the **unaligned** interface used by M0/M1, all-zero audio timestep 0 occurs in **0/3,395 train** and **0/728 valid** samples (fractions 0.0000 and 0.0000). Every timestep 0 has `P=1/O=1`. M0 retains it as an audio key/value by P; M1 retains it by P/O. Neither calls it a local missing event from its numeric value. As a cross-version reference only, the **aligned** A2 audio timestep 0 is all-zero in **3,395/3,395 train** and **728/728 valid**, with `P=2/O=2`: unresolved, not a verified or synthetic absence. Aligned arrays are not bridge inputs.

## Labels and native zero policy

| Split | Negative: `y<0` → 0 | Neutral: `y=0` → 1 | Positive: `y>0` → 2 | Sign mismatches |
|---|---:|---:|---:|---:|
| train | 967 | 758 | 1,670 | 0 |
| valid | 206 | 184 | 338 | 0 |

The regression target remains each sample's stored continuous `regression_label` (`y`, observed range −3 to 3); the bridge casts it to float32 for loss and does not substitute the class. The frozen [label audit](../../data_audit/label_audit.md) establishes that cleaning copied A2 labels unchanged.

M0 never constructs an O mask from feature magnitude. M1 reads the frozen tri-state P/O arrays; native all-zero rows with unresolved status remain `O=2` and are kept in attention. Synthetic S1–S6 gaps are separate copied interventions selected only from `P=1/O=1` eligible positions, with their own `O=0`; native data are not overwritten. No rule of the form `all(feature==0) ⇒ missing` appears in M0/M1.

## Future A3 correspondence

`NativeMulT.forward` takes arrays and P/O states, not a sample ID. The current **A2 training** runner does require A2 IDs for deterministic mask selection and prediction records; there is no final A3 runner yet. A future A3 runner could retain official file/order keys outside the model without inventing a sample-level ID, but that behavior is **not implemented or tested**. The frozen schema document was inspected; A3 sample files were not accessed.

This audit stops at interface findings. Existing completed bridge seeds remain as recorded evidence; no additional seeds, model edits, A3 predictions or promotion decision were made in this audit.
