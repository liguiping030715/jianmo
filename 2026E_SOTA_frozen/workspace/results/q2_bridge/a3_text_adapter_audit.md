# Q2 A3 text adapter validation

**Status: `BRIDGE_A3_TEXT_ADAPTER_VALIDATED`.** This is an A2-only interface validation for a future unaligned A3 runner. No model was trained, changed or tuned; no A2 test, A3 sample or A4 was accessed.

## Reusable adapter and pinned settings

[FrozenUnalignedTextAdapter](../../../q2_unaligned_text_adapter.py) implements `raw_text → BertTokenizerFast → [input_ids, attention_mask, token_type_ids] → frozen BertModel → 50×768 float32`. It loads only the pinned local `bert-base-uncased`, runs in eval/inference mode with no parameter gradients, and uses BERT batch size 16 and 8 CPU threads, matching the Bridge clean cache.

Tokenizer: `add_special_tokens=True`, `padding='max_length'`, `truncation=True`, `max_length=50`, right padding/truncation, lower case, explicit attention mask and token-type IDs. Special IDs: CLS 101, SEP 102, PAD 0, UNK 100. Triplet row order: `['input_ids', 'attention_mask', 'token_type_ids']`. `transformers==4.26.0`. Model revision `86b5e0934494bd15c9632b12f734a8a67f723594`.

| Frozen local file | SHA-256 |
|---|---|
| `pytorch_model.bin` | `097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a` |
| `vocab.txt` | `b49e80874c5efbc7f4cde245a812673b05ef1360ced56b8bc8c750eaeef3fe0f` |
| `tokenizer.json` | `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` |
| `tokenizer_config.json` | `a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10` |
| `config.json` | `b92c83fdd39b9dcdded83e388feefd12a9bfc6e4e81cda9328c73b6865b10b3f` |

## Token-level equivalence

| Split | Exact rows | Total | Mismatched rows | First mismatch |
|---|---:|---:|---:|---|
| train | 3395 | 3395 | 0 | none |
| valid | 728 | 728 | 0 | none |

## BERT hidden-state equivalence on every A2 valid row

Shape [728, 50, 768]; cache dtype `float32`, adapter dtype `float32`; both finite: True/True. Attention mask exact: True.

Max absolute error **0**; mean absolute error **0**; RMSE **0**; `np.allclose=True` with `rtol=1e-05`, `atol=1e-06`.

## End-to-end frozen M0 prediction equivalence

PATH A uses the frozen Bridge clean text cache. PATH B uses this adapter's raw-text representation. Both use identical A2-valid audio/vision, P/O and one unchanged checkpoint per seed.

| Seed | Max abs logit Δ | Mean abs logit Δ | Class disagreements | Max abs intensity Δ | Mean abs intensity Δ |
|---:|---:|---:|---:|---:|---:|
| 1729 | 0 | 0 | 0 | 0 | 0 |
| 2718 | 0 | 0 | 0 | 0 | 0 |
| 31415 | 0 | 0 | 0 | 0 | 0 |

| Seed | Path | Accuracy | macro-F1 | weighted-F1 | MAE | Pearson |
|---:|---|---:|---:|---:|---:|---:|
| 1729 | A cache | 0.633242 | 0.602956 | 0.628420 | 0.611057 | 0.635312 |
| 1729 | B raw text | 0.633242 | 0.602956 | 0.628420 | 0.611057 | 0.635312 |
| 2718 | A cache | 0.630495 | 0.587701 | 0.615409 | 0.578791 | 0.652265 |
| 2718 | B raw text | 0.630495 | 0.587701 | 0.615409 | 0.578791 | 0.652265 |
| 31415 | A cache | 0.615385 | 0.552925 | 0.587866 | 0.628077 | 0.634807 |
| 31415 | B raw text | 0.615385 | 0.552925 | 0.587866 | 0.628077 | 0.634807 |

## Mask and future runner contract

The regenerated attention mask is exactly the A2 `text_P` and the tokenizer-padding `text_O`. M0 pools only positions with `P!=0`; padding hidden states cannot enter its final pooled prediction. The adapter makes no observation decision from hidden-state magnitude and adds no M1 logic.

A future unaligned A3 runner takes an **external official file/order key**, raw text, audio `[500,74]`, vision `[500,35]` and audited audio/vision P/O. It uses this adapter to create text hidden states and `text_P`, then sends text/audio/vision with masks to the frozen M0, retaining the external key for output correspondence. `NativeMulT.forward` does not require an internal sample `id`. Aligned A3 `text_bert` must not be mixed with unaligned A/V. This contract was defined without loading or predicting any A3 sample.

The validated claim is numerical equivalence of the A2 raw-text route and Bridge route. It does not validate future A3 sample content, missingness, performance or file-order execution.
