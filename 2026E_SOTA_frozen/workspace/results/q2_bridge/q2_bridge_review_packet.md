# Q2 Bridge review packet

## Frozen results

- H11 M0 backbone competitiveness on A2 valid: **PASS**. Metric-gate label: `MULT_BACKBONE_SUPPORTED`.
- H12 observation adapter: **FAIL**. M1 is not promoted when this gate fails.
- Frozen E0 and E1 remain rejected; they were not retrained or tuned.
- R1 aligned baseline is contextual only. Its three-seed clean mean macro-F1 is 0.5725 and MAE 0.6072. Different input version prevents attributing any difference to backbone alone.

## Deployment boundary

The [read-only pitfall audit](pitfall_audit.md) remains `BRIDGE_TEXT_INTERFACE_INVALID` under the reviewer’s literal supplied-A3-`text_bert` requirement: same-version unaligned A3 supplies raw text rather than that field, and the current bridge has no A3 inference runner. A2 raw text reproduces its token triplets exactly, so a derived same-tokenizer pathway is plausible, but this bridge review does not silently substitute it for the specified field or mix aligned text with unaligned audio/vision. The A2 metric gate is therefore **not a final Q2 deployment approval**.

A [fixed-weight A2-valid interface probe](interface_probe.md) additionally recomputed all 728 valid BERT states from raw text with **zero** difference from the cache. On 32 valid rows, both seed1729 checkpoints gave zero logit/intensity difference between the two text routes; changing only padded BERT rows to 10,000 also produced zero output difference. This supports the derived unaligned raw-text path and padded-query output isolation. It does not create a supplied A3 `text_bert` field or validate an A3 inference runner.

Post-review interface validation: the reusable [unaligned text adapter](a3_text_adapter_audit.md) now reproduces A2 train/valid token triplets exactly and, on all 728 A2 valid samples, reproduces frozen BERT states and all three M0 checkpoints' predictions exactly. This resolves the *A2 raw-text-to-M0 interface implementation* concern. No A3 sample or full A3 runner has been executed; the earlier pitfall audit remains a record of the pre-adapter boundary.

## Evidence and restrictions

- All six planned completed runs have complete clean plus S1–S6 valid predictions, matching seed-specific native masks and checkpoint hashes. The failed first M0 attempt is retained in `q2_bridge_runs/M0_seed1729_attempt1_failed/`.
- No A2 test, A3, A4, or A1 input was used in the bridge training or metric summary. No official missing mask, physical time alignment, restoration, or Q3 faithfulness claim is made.
- S1–S6 couple modality, native index position and gap ratio. They screen model mechanisms only; they cannot estimate independent effects of those factors. Controlled factorial analysis remains for a later approved stage.

Details: [M0](M0_results.md), [M1 comparison](M1_vs_M0.md), [resources](resource_report.md), [machine decision](q2_bridge_decision.json).
