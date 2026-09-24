# AI / Tool Usage Log

Use this log when required by competition or institutional rules.

| Time | Tool / Model | Task | Input summary | Output actually used | Human verification / modification | Paper/code location |
|---|---|---|---|---|---|---|
| 2026-09-23 | Codex; Python/OpenCV FFmpeg backend | Stage 2.5 timestamp and interface evidence recovery | Official MP4s, frozen cleaning v2, official BERT model files | Container PTS/edit-list and decoded-frame cross-check; six Stage 2.5 contracts | Automated count/PTS checks on all 140 MP4s; model forward-pass and tokenizer checks; no labels or model training used | `workspace/evidence_recovery/` |
| 2026-09-23 | Hugging Face official `bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594` | Verify a common text encoder interface | Generic pretrained vocabulary and weights, official Attachment 2/3/4 text fields | Pinned tokenizer/encoder provenance and exact token-triplet compatibility audit | SHA-256 and finite encoder forward pass checked locally; no external sentiment training data | `workspace/references/bert-base-uncased/`; `workspace/evidence_recovery/text_interface.md` |
| 2026-09-23 | Codex | Stage 3 candidate design only | Official 2026E statement, frozen Stage 1–2.5 evidence, reviewer constraints | Four model cards, one model-agnostic Q3 wrapper, prespecified pilot matrix and promotion gates | Static cross-file/link and required-field checks; no training, checkpoint, A3/A4 selection, or new cleaning | `workspace/models/model_cards/`; `workspace/experiments/pilot_plan.md` |
