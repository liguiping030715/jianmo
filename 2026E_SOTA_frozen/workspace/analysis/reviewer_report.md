# Reviewer Report: Stage 2.5 Blocked-Gate Review

## Verdict

`READY_WITH_LIMITATIONS` for the profiler-only component scope defined in `workspace/analysis/pilot_scope_decision.md`.

The full Question 1 workflow remains blocked. In particular, model `a`, cross-sensor fusion, and evaluation of model `b` against model `a` are not approved.

## Critical issues

1. **Combined reference remains scientifically unsupported.** Microwave time zone, `QCflag=9`, transformed profile units, and shared vertical reference are unresolved. Any model-`a` pilot or profiler-versus-reference comparison would cross the evidence gate.
2. **No validated target exists for model `b`.** The permitted component pilot can test execution only; it cannot estimate accuracy or rank final approaches.

Required fix for expansion: obtain the exact clock/time-zone record, QC dictionary, transformed-export unit dictionary, and both sensors' vertical-reference definitions.

## Major issues

1. **Physical height is unavailable in the approved subset.** Sampling-height tokens may be used only as categorical gates; physical vertical gradients and flight-altitude claims are prohibited.
2. **The 30-minute window is inadequate for validation.** Six times per station permit only feasibility checks, not stability, generalization, or performance claims.
3. **`W6` has no resolved supplied-file unit.** It is excluded from the approved pilot.
4. **Station B is not approved for cross-sensor use.** Its two instrument coordinates differ by `32,739.660 m`, and its microwave cycles contain unexplained duplicates.

Required fix for expansion: supply height/datum evidence, a longer time window, the `Cn²` unit for this product, corrected station B metadata, and duplicate-cycle provenance.

## Minor issues

1. Supplied profiler filenames differ from the CAAC standard's prefix and station-code grammar. Internal metadata supports the approved parser, but filename semantics must not be generalized.
2. The five-digit profiler identifiers remain literal IDs rather than resolved four-letter airport codes.

Required fix: obtain organizer naming documentation before using these tokens as standardized site codes.

## Approval conditions

Reviewer approval applies only if all of the following are enforced:

- profiler files only;
- both stations kept separate;
- six stated UTC profiler times only;
- 33 stated common numeric gate tokens only, treated as categorical identifiers;
- `W1`-`W5` only;
- no microwave fields, `W6`, cross-sensor joins, physical-height claims, or performance claims;
- no silent imputation, row deletion, deduplication, or gate-level leakage.

No model design or experiment was reviewed or approved in this stage because none was requested or produced.

