# Question 1 Restricted Pilot Scope Decision

## Decision: READY_WITH_LIMITATIONS

A scientifically defensible **profiler-only component pilot** exists. It may test whether future candidate model-`b` calculations can run reproducibly on resolved wind-profiler inputs and produce a gate-indexed output. It may not construct model `a`, compare model `b` with model `a`, claim physical altitude, estimate turbulence accuracy, or answer Question 1 in full.

This decision supersedes the prior `BLOCKED` decision only for the exact scope below. All excluded Question 1 work remains blocked.

## Exact permitted data scope

### Stations

- `station_a` wind-profiler files: permitted.
- `station_b` wind-profiler files: permitted as a separate profiler site.
- The two stations must be analyzed separately. They may be compared descriptively only after preserving station identity; they must not be treated as collocated replicates.

### Sensor and files

Only the 12 files under:

- `workspace/data/raw/station_a/wind_profiler/`
- `workspace/data/raw/station_b/wind_profiler/`

are permitted.

Both microwave files are excluded from the pilot. They remain unchanged and available for later evidence recovery.

### Timestamps

Use exactly the six resolved profiler UTC observation-end times at each station:

- `2025-08-02 00:00:00`
- `2025-08-02 00:06:00`
- `2025-08-02 00:12:00`
- `2025-08-02 00:18:00`
- `2025-08-02 00:24:00`
- `2025-08-02 00:30:00`

No microwave timestamp may be joined to these times. No extra time point may be interpolated or fabricated.

### Sampling gates

Use exactly the 33 raw sampling-height tokens fully numeric at both stations and all six times:

`100, 160, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760, 820, 940, 1060, 1180, 1300, 1420, 1540, 1660, 1780, 1900, 2020, 2260, 2500, 2740, 2980, 3220, 3460, 3700, 3940, 4180, 4420`.

These must be labeled `sampling_gate_token` or equivalent. They may be sorted and matched by exact literal value, but may not be converted to metres/kilometres or labeled AGL/MSL.

Restricting to these 33 gates is an explicit balanced-subset rule, not silent missing-row deletion. All higher rows remain in the raw files and audit records; they are outside this pilot's permitted scope.

### Permitted measurement variables

| Existing parser field | Resolved meaning | Unit/convention | Permitted role |
|---|---|---|---|
| `W1` | Horizontal wind direction | degrees | Input or diagnostic; circular treatment must be addressed later during model design |
| `W2` | Horizontal wind speed | m/s | Input or diagnostic |
| `W3` | Vertical wind speed | m/s; downward positive, upward negative | Input or diagnostic |
| `W4` | Horizontal confidence | %, integer 0-100 | Quality/reliability input or stratification variable; no undocumented threshold |
| `W5` | Vertical confidence | %, integer 0-100 | Quality/reliability input or stratification variable; no undocumented threshold |

Permitted metadata are the literal station identifier, resolved profiler longitude/latitude, observation-field elevation above sea level in metres, radar type `LC`, UTC observation-end timestamp, and raw gate token.

### Explicitly excluded variables

- profiler `W6` / `Cn²`, because its supplied-file unit remains unresolved;
- every microwave profile code: `11`, `12`, `13`, and `14`;
- all microwave named scalar fields, including `SurTem`, `SurHum`, `SurPre`, `Tir`, `Rain`, `CloudBase`, `Vint`, and `Lqint`;
- microwave `QCflag` and every interpretation of value `9`;
- microwave timestamps, coordinates, final location metadata tokens, and station B duplicate-cycle occurrence IDs;
- any radial velocity, spectrum width, or SNR variable, because those are not documented fields in the supplied `ROBS` product.

Although microwave code `13` has a resolved physical meaning/unit, it is excluded because the current restricted pilot excludes the microwave sensor as a whole to avoid unresolved time, QC, and vertical-reference assumptions.

## Permitted pilot purpose

The pilot may determine only whether a future profiler-only candidate:

1. parses and preserves the permitted inputs reproducibly;
2. handles station and timestamp grouping without leakage;
3. operates independently at fixed sampling gates without requiring a physical height unit;
4. respects the resolved wind-direction, wind-speed, vertical-velocity, and confidence semantics;
5. produces finite, reproducible gate-indexed outputs for the permitted records;
6. exposes numerical failures, degenerate behavior, or excessive sensitivity within this small subset.

These permissions define an experiment boundary, not a selected formula or algorithm.

## Prohibited calculations and claims

The restricted pilot must not:

- construct or label any result as combined model `a`;
- calibrate, train, or validate model `b` against model `a`;
- claim that the gate tokens are metres, kilometres, AGL, or MSL;
- compute vertical derivatives or shear per unit physical height;
- use `W6` in a dimensional calculation;
- accept/reject microwave data based on `QCflag=9`;
- choose, merge, average, or delete station B duplicate microwave cycles;
- infer that profiler and microwave timestamp labels are simultaneous;
- report turbulence accuracy, operational thresholds, generalization, robustness, or final model superiority;
- treat gate rows as independent train/test samples;
- select the final turbulence indicator or final model from this pilot.

## Leakage and validation boundary

There are only six profiler times per station. If a later permitted pilot includes any holdout diagnostic, the indivisible sampling unit must be a complete `(station, UTC observation-end time)` profile. Gates from one profile may not be split across training and validation. With this sample size, such a holdout is only a code-path diagnostic and cannot support a performance claim.

## What remains blocked

The following remain blocked until the evidence listed in `blocker_review.md` is obtained:

- all model-`a` design and pilot work;
- all microwave-profiler fusion;
- all comparison or calibration of model `b` against model `a`;
- station B cross-sensor use;
- any physically referenced height profile;
- any unit-dependent use of microwave codes `11`, `12`, or `14`;
- any final/full Question 1 experiment or conclusion.

## Gate implication

The next stage, if separately requested, may begin candidate model design **only for the profiler-only component and only within the scope above**. This review does not itself begin model design.

**Final blocked-gate review decision: `READY_WITH_LIMITATIONS`**

