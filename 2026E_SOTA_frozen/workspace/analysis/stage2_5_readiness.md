# Stage 2.5 Readiness Assessment

## Decision: BLOCKED

Stage 2.5 substantially resolves the wind-profiler schema and the physical meanings of microwave profile codes, but critical metadata still prevents a scientifically defensible Question 1 pilot. The project therefore remains at the evidence-recovery gate and must not proceed to model design.

## What is now clear

1. The supplied profiler product is structurally an AP-117-TM-2013-01 `ROBS` real-time sampling-height product.
2. Profiler `W1`-`W6` are, in order: horizontal wind direction, horizontal wind speed, vertical wind speed, horizontal confidence, vertical confidence, and vertical-direction `Cn²`.
3. Wind units/sign conventions and confidence units/ranges are defined by the official standard.
4. Slash-only profiler groups are the official width-preserving missing representation.
5. Profiler timestamps denote observation end time in UTC.
6. Profiler metadata values `16.2` and `40.6` are observation-field elevations above sea level in metres.
7. Microwave codes `11`, `12`, `13`, and `14` denote temperature, vapor-density, relative-humidity, and liquid-water-density profiles.
8. `SurTem`, `SurHum`, `SurPre`, `Tir`, `Rain`, `CloudBase`, `Vint`, and `Lqint` have externally supported physical meanings; several supplied units are also literal in the headers.
9. The microwave timestamp is an observation-set completion/end time in the matched manufacturer format.
10. The supplied `ROBS` product does not contain documented spectrum width or radial velocity; those belong to a different profiler product class.

## What remains uncertain

### Critical blockers

| Blocker | Why it prevents a defensible pilot | Minimum evidence needed |
|---|---|---|
| Microwave time zone | Profiler times are UTC, but the radiometer manual only says time comes from the host clock and recommends UTC sync. Equal labels do not prove simultaneous measurements. | Organizer/site confirmation or original acquisition configuration/log showing the radiometer clock/time zone |
| `QCflag=9` meaning | Every microwave row has the same unresolved QC token. Without its definition, all rows might be valid, suspect, or invalid; no safe validity filter exists. | Exact export-system/manufacturer/organizer QC code table |
| Units of transformed profile codes `11`, `12`, `14` | The exact code dictionary gives native units, but code `11` visibly underwent a conversion in this export. Unit-dependent thermodynamic calculations would be unsupported. | Export-specific data dictionary or organizer confirmation of post-export units |
| Vertical reference | Profiler Table 17 does not explicitly state the sampling-height unit/datum, and microwave height datum is not documented. Vertical derivatives and cross-sensor pairing remain ambiguous. | Exact A/B height definition for both exports, including unit and AGL/MSL reference |

### Station B blockers

| Blocker | Reproduced fact | Consequence |
|---|---:|---|
| Coordinate discrepancy | `32,739.660 m` between microwave and profiler coordinates | Station B cannot be treated as a collocated combined-instrument reference |
| Duplicate microwave cycles | six repeated timestamp labels with distinct cycle payloads | No justified selection, aggregation, or revision ordering exists |

### Nonconformance requiring caution

The CAAC filename standard expects `Z_RADR_I_` and a four-letter airport code. Supplied files use `Z_RADA_I_` and five-digit numeric identifiers. The internal `WNDROBS`/metadata/`ROBS` structure is an exact match and supports field resolution, but filename semantics beyond the matching timestamp/product positions should not be generalized.

## Why this is not `READY_WITH_LIMITATIONS`

Excluding station B would remove the large coordinate discrepancy and duplicate-cycle problem, but it would not solve the station A microwave time zone, the universal `QCflag=9`, the transformed profile units, or the two instruments' vertical references. Those uncertainties affect whether station A measurements are simultaneous, valid, and vertically commensurate—the minimum conditions for constructing the combined reference required by Question 1 model `a`.

A profiler-only pilot of some future candidate may be structurally possible because the six `ROBS` fields are now identified. However, Question 1 requires comparison against a combined profiler-radiometer reference. A model-`b` pilot without a defensible model-`a` target would not satisfy the requested experiment. This report therefore does not split the task to force a partial readiness decision.

## Data checks required before the gate can reopen

1. Obtain the organizer/export-system dictionary for `QCflag`, including the value `9` and whether it applies per row, profile, or cycle.
2. Obtain the actual radiometer acquisition time zone or clock configuration for 2025-08-02 and verify it against the profiler's UTC timestamps.
3. Obtain the unit mapping for the transformed 83-level profiles, especially codes `11`, `12`, and `14`.
4. Obtain explicit height-unit and vertical-datum definitions for both sensors; then recompute common height support without a conditional correspondence.
5. Correct or explain station B's coordinate pair. If the devices are not collocated, formally exclude station B from any combined-instrument experiment.
6. Define the provenance/order of station B duplicate cycles and whether they are independent acquisitions, retransmissions, or revisions. Do not delete any cycle during investigation.
7. Confirm whether the five-digit profiler identifiers are WMO/station identifiers or another organizer-specific code; retain them as literal IDs until confirmed.
8. Confirm that the 30-minute package is intentionally a code-path rehearsal subset and not evidence for full-window performance claims.

## What can continue while blocked

Only metadata-preserving work is safe: retain the resolved aliases in documentation, keep the current strict parsers, preserve all duplicate cycles, and request the minimum evidence above. No turbulence formula, model candidate, calibration, interpolation, vertical derivative, or performance estimate should be started.

## Stage boundary

Stage 2.5 stops here. No model has been selected or designed, no data have been filtered or aligned for training, and no numerical turbulence result has been produced.

**Final Stage 2.5 decision: `BLOCKED`**
