# Fixed-weight A2-valid text interface probe

All 728 A2 valid `raw_text` strings reproduced the stored `text_bert` token triplets with the pinned tokenizer. Re-encoding with the same frozen local BERT produced [728,50,768] states. No model weight was changed and no A3 sample was opened.

Maximum absolute difference from the frozen clean BERT cache: **0**.

On the first 32 valid rows, fixed seed1729 M0/M1 checkpoints gave the following differences. Padded text BERT states were also replaced by 10,000 solely in a copied input to test whether padded queries can affect the final prediction.

| Checkpoint | Logit difference: raw text vs cache | Intensity difference | Logit difference: padding stress | Intensity difference |
|---|---:|---:|---:|---:|
| M0 | 0 | 0 | 0 | 0 |
| M1 | 0 | 0 | 0 | 0 |

Padded positions in the 32-row probe batch: 884.

This validates an A2 shadow path `raw_text → same tokenizer → same BERT → same model output` and shows that query-side padding computation does not influence these fixed checkpoints. The strict supplied-A3-`text_bert` requirement remains unresolved for the unaligned version, and there is still no A3 inference runner. The earlier [pitfall audit](pitfall_audit.md) is preserved as the review boundary.
