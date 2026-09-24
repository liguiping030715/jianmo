# Stage 2.5 blocker register

Evidence levels follow `.agents/skills/evidence-recovery/SKILL.md`: the official statement and supplied file content are Level A for what is actually present; numeric-pattern interpretations without format documentation remain inference, not resolved semantics.

| ID | File/field | Current evidence | Status and modeling consequence | Needed to resolve |
|---|---|---|---|---|
| ER-01 | Attachment 3 `text_bert` aligned | All 30 arrays finite, integral float32, exact int64 cast; pinned BERT vocabulary/weights loaded; [text interface](text_interface.md) | **RESOLVED executable token-to-encoder path**. The 131 active `[UNK]` positions have unresolved observation/source meaning and remain `O=2`, not missing truth. | Official A3 text intervention metadata would resolve semantic status, but is not required for a restricted pilot. |
| ER-02 | Attachment 3 unaligned `raw_text` | Present in all 30; the same pinned tokenizer exactly reproduces all supplied A2 and A4 token triplets; local BERT forward pass passed | **RESOLVED executable raw-text-to-encoder path**; no precomputed numeric `text` is needed or assumed | Keep same-version raw text and tokenizer; do not import paired aligned A3 tokens. |
| ER-03 | Attachment 2 unaligned `vision_lengths` | Nonzero vectors beyond listed length in 618 train, 141 valid, 131 test | **UNRESOLVED field semantics**; prefix truncation unsafe. Tri-state interface retains unknown positions. | Official generation rule matching this exact PKL. |
| ER-04 | Attachment 3 local zero intervals | No explicit original ID, `P` or `O`; multiple internal and terminal zero runs | **UNRESOLVED exact missing-mask semantics**; `zero_context=3` is diagnostic only. Restricted tri-state pilot is feasible. | Official missingness-generation metadata for stronger claims. |
| ER-05 | Attachment 1/4 MP4 timebase | Container `stts/ctts/elst` and FFmpeg-backed decoded PTS agree in all 140 files; [timebase audit](media_timebase.md) | **RESOLVED video frame PTS, edit-list presentation duration, audio packet PTS**. `mvhd` duration alone overstates playable duration in many files. | Decoded PCM audio timestamps if that resolution is needed later. |
| ER-06 | Feature/word-to-media timestamps | No PKL extractor timestamp table or word alignment; media frames/packets now timestamped | **UNRESOLVED existing feature-row and spoken-word seconds**; report feature/token interval and null mapping with uncertainty. | Official extraction/alignment log or new logged Q1 extraction for newly made features. |
| ER-07 | Reliability `R` | No quality field in supplied PKLs | **UNRESOLVED as input metadata**; may later be a train-fitted model estimate | Do not invent a ground-truth reliability label |

No inferred Level D pattern is promoted to an official field definition. The [readiness reassessment](stage2_5_readiness.md) permits only a restricted pilot interface with the listed limitations; this request stops at Stage 2.5.
