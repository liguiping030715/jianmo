# Blocked-Gate Review: Question 1 Pilot Feasibility

## Review scope

This is an independent review of the Stage 2.5 `BLOCKED` decision. It asks a narrower question than the prior readiness report: whether each unresolved item prevents **every** scientifically defensible Question 1 pilot, or whether it can be avoided by an explicit restriction of station, sensor, field, time, or calculation.

This review does not reinterpret an unknown field, select a turbulence indicator, design a model, align microwave and profiler records, or alter the raw data. A “pilot” here means a small, reproducible component experiment intended to test feasibility. It does not mean a complete answer to Question 1.

## Classification criteria

- `CRITICAL`: no defensible Question 1 pilot can proceed until the issue is resolved.
- `SCOPE_LIMITING`: a valid component pilot can proceed only if the affected station, sensor, field, time range, or calculation is explicitly excluded.
- `NONCRITICAL_FOR_PILOT`: the issue blocks final/full modeling or strong claims, but not a narrowly defined pilot.

## Summary classification

| ID | Unresolved issue | Classification | Consequence for the permitted pilot |
|---|---|---|---|
| B1 | Microwave time zone / timestamp semantics | `SCOPE_LIMITING` | Exclude all microwave data and all cross-sensor temporal pairing |
| B2 | Meaning of `QCflag=9` | `SCOPE_LIMITING` | Exclude all microwave data rather than accepting or rejecting rows by an unknown flag |
| B3 | Unresolved units for microwave profile fields | `SCOPE_LIMITING` | Exclude microwave profile codes `11`, `12`, and `14`; selected pilot excludes the whole microwave sensor |
| B4 | Vertical datum / height reference | `SCOPE_LIMITING` | Use only fixed raw profiler sampling-gate tokens as categorical gate identifiers; prohibit physical-height claims and vertical derivatives |
| B5 | Station B approximately 32.74 km coordinate conflict | `SCOPE_LIMITING` | Prohibit station B cross-sensor fusion; station B profiler-only records may be used as their own site |
| B6 | Station B duplicate microwave cycles | `SCOPE_LIMITING` | Exclude station B microwave data; preserve every duplicate in raw/audit files |
| B7 | Short 30-minute observation window | `NONCRITICAL_FOR_PILOT` | Permit only execution/feasibility checks; prohibit performance, stability, and generalization claims |

No reviewed issue is `CRITICAL` to **every possible component pilot**. Several remain critical to a combined profiler-radiometer pilot of model `a` and to any comparison of model `b` against model `a`.

## B1. Microwave-radiometer time zone / timestamp semantics

**Classification: `SCOPE_LIMITING`**

### Calculation affected

This issue affects every calculation that pairs microwave cycles with profiler observations, including construction of a contemporaneous combined reference profile, calibration of profiler-only results against that reference, and any time-lag or temporal-change calculation across sensors.

### Why

The microwave manufacturer evidence supports completion/end-time semantics, and the profiler standard supports UTC observation-end time. The microwave time zone is still unknown because its clock derives from the host operating system; a recommendation to synchronize to UTC does not prove the supplied system used UTC. Equal timestamp labels therefore do not prove simultaneous physical observations.

### Station impact

- **station_a:** affected. It has six exact label matches, but label equality is not proof of common time zone.
- **station_b:** affected, with the additional complication that only five profiler labels have exact microwave matches and duplicates exist.

### Scope restriction that avoids the issue

Exclude both microwave files and prohibit every cross-sensor join. Use only profiler timestamps, whose UTC/end-time meaning is resolved. No assumption about the microwave clock is then required.

### Evidence needed for full resolution

An organizer/site record or original acquisition configuration/log showing the radiometer clock setting and time zone on 2025-08-02, plus confirmation that the exported `DateTime` retained that convention.

## B2. Meaning of `QCflag=9`

**Classification: `SCOPE_LIMITING`**

### Calculation affected

This affects acceptance, rejection, weighting, or uncertainty labeling of every microwave profile row. It therefore affects any model-`a` input and any validation target derived from those rows.

### Why

All supplied microwave rows have literal value `9`, but no A/B source defines that value. Treating it as valid, invalid, or a bit mask would invent semantics. Because the value is universal, filtering by an assumed interpretation could either admit all questionable data or discard the entire microwave dataset.

### Station impact

- **station_a:** affected on all 64 microwave rows.
- **station_b:** affected on all 72 microwave rows.

The profiler confidence fields are separately defined by the CAAC standard and are not affected by the microwave `QCflag` uncertainty.

### Scope restriction that avoids the issue

Exclude all microwave data from the pilot. Do not delete or relabel its rows; they remain preserved for later recovery. The permitted profiler-only scope may use the resolved profiler confidence fields `W4` and `W5` without transferring any meaning from `QCflag`.

### Evidence needed for full resolution

The exact organizer/export-system or manufacturer QC dictionary defining the `QCflag` column, the code `9`, its scope (row/profile/cycle), and any associated validity action.

## B3. Unresolved units for microwave profile fields

**Classification: `SCOPE_LIMITING`**

### Calculation affected

The unresolved supplied-export units of code `11` (temperature), `12` (vapor density), and `14` (liquid-water density) affect thermodynamic gradients, stability quantities, water-content transformations, dimensional thresholds, and any fused metric using their magnitudes. Code `13` relative humidity is resolved as percent, but that does not repair the other fields.

### Why

The manufacturer dictionary resolves physical meanings and native units, but the supplied export is transformed. In particular, code `11` is not in the manufacturer's native Kelvin representation. Applying native units to the transformed file would be unsupported.

### Station impact

- **station_a:** codes `11`, `12`, and `14` are affected.
- **station_b:** the same codes are affected; station B additionally has degraded temperature-unit glyphs in named fields.

### Scope restriction that avoids the issue

For the selected pilot, exclude the entire microwave sensor. A less restrictive future pilot could use only fields with resolved export units, but it would still remain blocked by B1, B2, and B4 for cross-sensor fusion. No current pilot may use codes `11`, `12`, or `14` in a unit-dependent calculation.

### Evidence needed for full resolution

The organizer's transformed-export dictionary or export software documentation giving post-export units and any conversions for every profile code and scalar field.

## B4. Vertical datum / height reference

**Classification: `SCOPE_LIMITING`**

### Calculation affected

This affects cross-sensor height matching, conversion to altitude above mean sea level or height above ground, vertical derivatives, vertical shear per unit distance, interpolation between sensors, and claims that an output occurs at a physical flight altitude.

### Why

The profiler first field is authoritatively identified as sampling height, but its unit and AGL/MSL datum are not explicitly defined in the matched Table 17 evidence. Microwave columns print kilometre-valued levels, but their datum is undocumented. Numerically corresponding tokens cannot establish a shared physical vertical coordinate.

### Station impact

- **station_a:** affected for physical-height interpretation and all profiler-microwave height alignment.
- **station_b:** affected in the same way, independently of its horizontal coordinate conflict.

### Scope restriction that avoids the issue

Use profiler sampling-height tokens only as **opaque, ordered gate identifiers**. Restrict the pilot to calculations performed independently at each fixed gate through time. Prohibit:

- conversion of a token to metres, kilometres, AGL, or MSL;
- finite differences or derivatives divided by vertical separation;
- interpolation between gates based on physical distance;
- microwave-profiler height matching;
- flight-altitude or vertical-extent claims.

For a balanced, missing-free component pilot, use the 33 raw gate tokens fully numeric at both stations and all six times:

`100, 160, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760, 820, 940, 1060, 1180, 1300, 1420, 1540, 1660, 1780, 1900, 2020, 2260, 2500, 2740, 2980, 3220, 3460, 3700, 3940, 4180, 4420`.

These numbers are literal gate tokens, not asserted metric heights.

### Evidence needed for full resolution

An A/B source explicitly tying the `ROBS` sampling-height field to a unit and vertical datum, and equivalent documentation for the microwave export levels. Site-elevation/reference rules must state how to place both sensors in one vertical coordinate.

## B5. Station B approximately 32.74 km coordinate conflict

**Classification: `SCOPE_LIMITING`**

### Calculation affected

This affects any station B claim of collocation, any combined station B model `a`, and any use of station B microwave data as contemporaneous local thermodynamic context for station B profiler data.

### Why

Stage 2 reproducibly derived a `32,739.660 m` great-circle separation between the two supplied station B coordinate pairs. This is not approximate instrument collocation at a single site. No evidence identifies which coordinate is erroneous or whether the files intentionally represent different sites.

### Station impact

- **station_a:** unaffected by this blocker; its supplied coordinate separation is `2.425 m`.
- **station_b:** affected only for cross-sensor fusion. The profiler file's own internally consistent coordinate metadata may still identify the profiler-only series as one site.

### Scope restriction that avoids the issue

Do not fuse station B microwave and profiler data. In the selected profiler-only pilot, station B may be analyzed as a separate profiler site using only its profiler metadata and observations. Do not pool station A and B as collocated observations or use either as the other's ground truth.

### Evidence needed for full resolution

An organizer station inventory, corrected file, or instrument deployment log identifying the intended coordinates of both station B instruments and whether they were meant to be collocated.

## B6. Station B duplicate microwave cycles

**Classification: `SCOPE_LIMITING`**

### Calculation affected

This affects station B microwave time-series construction, cycle selection, aggregation, temporal weighting, and cross-sensor matching. Selecting one cycle or averaging the pair without provenance could discard a revision or merge independent acquisitions.

### Why

Six station B timestamp labels occur twice, and their cycles are distinct rather than exact duplicates. No source defines whether they are retransmissions, revisions, separate acquisitions, or concatenated runs.

### Station impact

- **station_a:** unaffected; no duplicate microwave timestamp cycles occur.
- **station_b:** affected for all uses of duplicated microwave times. Its profiler files themselves are unique and strictly ordered.

### Scope restriction that avoids the issue

Exclude station B microwave data. Preserve all duplicate cycles unchanged. Station B profiler-only records remain permitted because this blocker does not occur in the profiler files.

### Evidence needed for full resolution

Acquisition/export logs or organizer documentation defining cycle identity, sequence, retransmission/revision behavior, and the authoritative rule for retaining or aggregating repeated labels.

## B7. Short 30-minute observation window

**Classification: `NONCRITICAL_FOR_PILOT`**

### Calculation affected

The short window affects estimation stability, temporal-scale selection, train/validation splitting, uncertainty assessment, generalization, rare-event coverage, and any performance comparison intended to represent the full competition interval.

### Why

Each profiler supplies only six UTC observation-end times from 00:00 through 00:30 at six-minute spacing. This is inadequate for robust temporal validation or final performance claims. A pilot, however, may legitimately be a small execution and feasibility test if its conclusions are correspondingly narrow.

### Station impact

- **station_a:** affected; six profiler times.
- **station_b:** affected; six profiler times.

### Scope restriction that avoids invalid claims

Use all six profiler times only for deterministic pipeline execution, missingness handling, numerical sanity checks, and checking whether a candidate profiler-only calculation can produce a gate-indexed output. Do not:

- estimate final performance or generalization;
- optimize many parameters on these six times;
- perform random row/height splits;
- describe the interval as representative of the stated three-hour/full dataset;
- select a final model from apparent pilot performance.

Whole UTC profiles—not individual gates—must remain the indivisible temporal unit in any later chronological holdout or leave-one-time diagnostic, to avoid leakage.

### Evidence needed for full resolution

The full intended observation window, or organizer confirmation that the 30-minute subset is only a rehearsal package. Substantive validation also requires enough independent times/events to support the chosen metrics.

## Reviewer synthesis

The prior `BLOCKED` decision was correct for a **combined model-`a` pilot and model-`b` comparison**, because the unresolved microwave time, QC, units, and vertical reference are jointly fatal to that scope. It was broader than necessary for a **profiler-only component pilot**.

A defensible restricted pilot exists because the official standard resolves profiler field meanings, units for `W1`-`W5`, confidence semantics, missing-value encoding, and UTC/end-time semantics. By excluding all microwave data, excluding unitless `W6`, treating sampling-height tokens only as categorical gates, and limiting claims to execution feasibility, the pilot does not rely on any unresolved field interpretation.

This conclusion does not imply that model `a` is ready, that model `b` can be validated, or that Question 1 can be answered from the 30-minute subset.

