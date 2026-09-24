# Stage 2 Readiness Assessment: Question 1 Rehearsal Subset

## Decision: BLOCKED

The data are **structurally ready for repeatable parsing and audit**, but Question 1 is **blocked from a defensible pilot of model `a` or model `b`**. The block is not caused merely by small sample size: the supplied files do not define the physical meanings and units of the profiler fields needed for a turbulence calculation, and station B's two instrument coordinates differ by about 32.74 km. Proceeding would require fabricated semantics or an unsupported collocation assumption, both prohibited.

This decision does not select a turbulence indicator or algorithm and does not begin model design.

## What is already clear

1. The scoped raw package contains 16 files: 14 meteorological data files and two `.DS_Store` files.
2. Both stations supply one microwave table and six wind-profiler files over a 30-minute interval on 2025-08-02.
3. All 14 data files are structurally parseable with strict width, marker, numeric-token, timestamp, and filename/metadata checks.
4. Microwave tables have 95 columns with eight named scalar fields, 83 labeled height fields, four undocumented code rows per cycle, and a `QCflag` field.
5. Profiler files have 62 vertical rows, each containing one vertical-coordinate token and six undocumented fields retained as `W1`–`W6`.
6. Profiler upper-level missingness is represented by slash-only rows and is contiguous above the last numeric row.
7. Station A offers six exact common microwave/profiler timestamps. Station B offers five exact common timestamp labels; 00:24 has two microwave cycles and 00:00 has no exact microwave label.
8. Under the statement-supported but file-unconfirmed metre correspondence, both sensors have six exact common vertical tokens: 100, 400, 700, 1,300, 1,900, and 2,500.
9. Station A's microwave and profiler coordinates differ by a derived 2.425 m. Station B's differ by 32,739.660 m.
10. Station B's six duplicated microwave timestamp labels contain distinct cycles and must be preserved until their acquisition semantics are known.
11. The complete audit is reproducible and confirms that raw-file hashes are unchanged.

## What can be used safely now

| Item | Safe current use |
|---|---|
| File paths, sizes, hashes, encodings | Provenance and integrity checks |
| Timestamps | Exact-label inventory and chronology checks only |
| Microwave labeled height grid | Grid inventory in the units printed in its header |
| Profiler vertical tokens | Ordered-grid and missingness inventory; metre interpretation remains conditional |
| Microwave named fields | Retain under their literal header names; station B's degraded unit strings remain unresolved |
| `q11`–`q14` | Opaque numeric arrays and cross-field consistency checks only |
| `W1`–`W6` | Opaque numeric arrays, range/variation checks, and missing masks only |
| `QCflag` | Preserve/count the literal token `9`; do not filter on it |
| Duplicate station B cycles | Preserve separately with occurrence IDs |
| Station A coordinate pair | Evidence of horizontal agreement at displayed precision |

These uses support data engineering and metadata resolution, not a turbulence estimate.

## What remains uncertain or unusable for modeling

| Blocking uncertainty | Why it matters |
|---|---|
| Meanings and units of profiler `W1`–`W6` | It is impossible to verify that wind, radial velocity, spectrum width, or other turbulence-relevant inputs are present |
| Meanings and units of microwave codes `11`–`14` | Numerical coincidences cannot substitute for a data dictionary; `q13` behaves differently between stations |
| Meaning of `QCflag=9` | Validity filtering cannot be performed safely |
| Profiler vertical-coordinate unit and AGL/MSL datum | Vertical derivatives and microwave/profiler alignment would be ambiguous |
| Meaning/unit of site metadata values `6.5`, `9.4`, `16.2`, and `40.6` | They cannot be used as elevations or vertical offsets |
| Station B coordinate conflict | A combined-instrument profile cannot be claimed at one site |
| Station B duplicate-cycle semantics | No justified choice exists among retaining, selecting, or aggregating repeats for a model |
| Observation timestamp convention and time zone | Exact labels are available, but interval meaning and near-time matching are unresolved |
| Only 30 minutes and six profiler times | Robust temporal validation, generalization assessment, and full-window claims are unavailable |
| No direct turbulence truth | A future reference from model `a` would itself be model-derived, not observed ground truth |

## Why the data cannot yet support model `a`

A combined profiler–radiometer reference requires identified physical variables, valid QC, a justified vertical/time match, and spatial collocation. Station A satisfies only the structural and horizontal-coordinate portions. Station B additionally fails the collocation requirement. Thus, no defensible combined reference target can be constructed from the supplied files alone.

## Why the data cannot yet support model `b`

The profiler-only data are structurally complete at lower levels, but the six measurement fields have no supplied names or units. It is therefore impossible to determine whether the observations needed by any candidate turbulence indicator are available. Without a valid model-`a` reference, there is also no supported calibration target for comparing a future profiler-only method.

## Checks or information required before Stage 3

The minimum unblock set is:

1. Obtain an authoritative field dictionary for the `WNDROBS 01.20` / `ROBS` product, including the order, name, unit, missing-value convention, and quality meaning of all six values.
2. Obtain the microwave code mapping for `11`–`14`, including units and the definition of `QCflag=9`.
3. Confirm profiler vertical-coordinate units and datum, microwave height datum, and the meanings of the two instruments' site metadata numbers.
4. Correct or authoritatively explain the station B coordinate pair. If one instrument is from a different site, do not use B as a collocated pair.
5. Explain the repeated station B microwave timestamp cycles and the two out-of-order blocks; define whether they are separate samples, retransmissions, revisions, or another product behavior.
6. Confirm timestamp time zone and whether labels denote instantaneous time, interval start, interval end, or retrieval time.
7. Confirm whether this 30-minute subset is intentionally sufficient only for a code-path rehearsal or supply the stated analysis window for any substantive temporal experiment.
8. Before any later experiment, define split units as whole chronological cycles/profiles—not individual code rows or heights—to prevent leakage.

## Reusable Stage 2 deliverables

- `workspace/analysis/scripts/question1_parsers.py`: strict, reusable parsers that preserve opaque fields.
- `workspace/analysis/scripts/run_stage2_audit.py`: end-to-end read-only audit with raw hash verification.
- `workspace/analysis/scripts/README.md`: execution instructions.
- `workspace/analysis/audit_artifacts/audit_summary.json`: complete nested summary.
- `workspace/analysis/audit_artifacts/*.csv`: file, profiler, duplicate-cycle, time-alignment, and height-alignment evidence.

## Stage boundary

Stage 2 ends here. No turbulence formula, final indicator, model architecture, fitted parameter, aligned training table, or numerical turbulence result has been produced.
