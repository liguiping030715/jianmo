# M0 final architecture acceptance audit — fixed protocol

This protocol is fixed before the new inference calls. Inputs are only frozen A2 valid `unaligned_500` arrays and frozen clean BERT cache, with M0 checkpoints 1729/2718/31415. No training, parameter changes, A2 test, A3 or A4 access. The executed `q2_bridge.py` SHA-256 must equal `dc0367156de2bd7108d7910557692c173c35964b5c2307beb0dd8d32a2fedccf` in all three run environment records. The three checkpoint hashes must match their saved `metrics.json` values. The untouched FULL predictions must reproduce each saved clean valid metric within 1e-6; otherwise the audit is blocked.

## Conditions

- For within-sample temporal permutation or reversal, eligible native positions are exactly those with frozen `P=1` and `O=1`. Permute only their feature rows within that same sample and modality. Leave `P/O`, all other feature rows, text, labels, and sample order unchanged. `P=2/O=2` positions remain unresolved and untouched. Reverse the eligible row sequence for the reversal conditions.
- Deterministic permutation indices use SHA-256 of `m0-order-v1|sample_id|modality`, interpreted as the first 8 bytes of an unsigned seed for NumPy's `default_rng`. The resulting features are applied to each of the three checkpoints identically.
- For cross-sample audio and vision permutations, independently seed NumPy with SHA-256 of `m0-donor-v1|modality`. Make one derangement per modality by cyclically shifting a shuffled 728-row index. Every receiver gets a different valid donor. Copy the donor's full 500-step feature sequence **and its P/O masks**; keep receiver text and label. Audio and vision donor mappings are shared across all checkpoints and written once to `m0_permutation_manifest.jsonl`. The combined condition applies both independent mappings.
- Batch size 64, eval mode, no dropout, fixed checkpoint. Compute Accuracy, macro-F1, weighted-F1, MAE, Pearson on all 728 receiver labels. Save absolute max and elementwise mean logit changes, and absolute max/mean intensity changes from FULL.

## Decision rules

- Numerical equivalence: max absolute logit **and** intensity change ≤1e-5. Effective native order sensitivity for an A/V branch requires at least one of its two within-sample order conditions to have mean absolute logit change >1e-4 **or** mean absolute intensity change >1e-4 in each of the three seeds. The branch is described as order sensitive only if this is met. No result-based relaxation.
- Sample-specific predictive influence: for a modality's donor-permuted condition, at least 2/3 seeds must each have max logit change >1e-5 and either macro-F1 drop ≥0.005, weighted-F1 drop ≥0.005, MAE increase ≥0.005, or Pearson drop ≥0.005 versus FULL. This is a validation diagnostic, not a causal effect. At least one of audio or vision must satisfy this for the multimodal claim. The combined condition is descriptive, not an alternative rescue gate.
- Structural gate: actual projections, dimensions, positional buffers, residuals, LayerNorm, FFN, dropout and masks must be dimensionally valid. The all-masked key fallback must avoid nonfinite outputs. The architecture name must reflect only implemented directions.
- Status precedence: technical/source/reproduction failure → `M0_ARCHITECTURE_AUDIT_BLOCKED`; otherwise a failed claimed-order gate → `M0_MISSING_EFFECTIVE_POSITION_ENCODING`; otherwise failed sample-specific modality gate → `M0_MULTIMODAL_USE_NOT_SUPPORTED`; otherwise → `M0_ARCHITECTURE_ACCEPTED`.

The prior unaligned A3 raw-text adapter limitation is carried forward in the report and does not cause an A3 access or override this architecture-only decision.
