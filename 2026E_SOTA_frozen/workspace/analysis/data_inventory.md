# Stage 2 Data Inventory: Question 1 Rehearsal Subset

## Scope and evidence policy

This inventory covers only:

- `workspace/data/raw/station_a/`
- `workspace/data/raw/station_b/`

It does not inspect or use data for Questions 2 or 3. The raw files were read only. The audit compared the SHA-256 map of all 16 files before and after parsing and found the maps identical.

All undocumented fields remain opaque:

- microwave profile codes are called `q11`, `q12`, `q13`, and `q14`;
- the six profiler values following the first vertical-coordinate token are called `W1` through `W6`;
- `QCflag` values are counted but not interpreted;
- the fourth profiler metadata value and the final microwave location-line value are retained without assigning meaning or units.

Reproduce the inventory from the repository root with:

```powershell
python workspace/analysis/scripts/run_stage2_audit.py
```

The canonical machine-readable inventory, including every SHA-256 digest, is `workspace/analysis/audit_artifacts/file_inventory.csv`. The full nested audit is `workspace/analysis/audit_artifacts/audit_summary.json`.

## Complete file inventory

There are 16 files totaling 122,826 bytes: 2 microwave-radiometer text files, 12 wind-profiler text files, and 2 macOS `.DS_Store` filesystem-metadata files. Fourteen files contain meteorological records.

| Station | Relative path under `workspace/data/raw` | Classification | Extension | Bytes |
|---|---|---|---:|---:|
| A | `station_a/.DS_Store` | Filesystem metadata | none | 6,148 |
| A | `station_a/microwave_radiometer.txt` | Microwave-radiometer table | `.txt` | 35,493 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802000000_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802000600_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802001200_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802001800_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802002400_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| A | `station_a/wind_profiler/Z_RADA_I_58235_20250802003000_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/.DS_Store` | Filesystem metadata | none | 6,148 |
| B | `station_b/microwave_radiometer.txt` | Microwave-radiometer table | `.txt` | 42,097 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802000000_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802000600_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802001200_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802001800_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802002400_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |
| B | `station_b/wind_profiler/Z_RADA_I_58238_20250802003000_P_WPRD_LC_ROBS.TXT` | Wind-profiler `ROBS` text | `.TXT` | 2,745 |

## Directory and sensor organization

Each station package has the same organization:

```text
station_x/
├── .DS_Store
├── microwave_radiometer.txt
└── wind_profiler/
    └── six timestamped *_P_WPRD_LC_ROBS.TXT files
```

The profiler files occur at 00:00, 00:06, 00:12, 00:18, 00:24, and 00:30 on 2025-08-02 at both stations. The supplied data therefore cover 30 minutes, not the three-hour interval described for the full competition problem. No time-zone declaration is present in the inspected files.

## Microwave-radiometer tables

### Shared schema

Both files have two non-data lines followed by tab-separated records:

1. a location/metadata line;
2. a 95-column header;
3. data rows.

The 95 columns comprise:

- `Record`, `DateTime`, and the undocumented code column labeled `10`;
- eight named scalar fields: surface temperature, surface humidity, surface pressure, `Tir`, `Rain`, `CloudBase`, `Vint`, and `Lqint` as represented by each file's literal header text;
- 83 labeled vertical columns from `0.000(km)` through `10.000(km)`;
- `QCflag`.

All data rows have exactly 95 fields and all tokens in the named and vertical numeric positions parse as finite numbers. No blank field occurs. The tables do not end with a newline.

The 83-height grid is identical at both stations:

| Labeled interval | Labeled increment | Levels | Adjacent steps |
|---|---:|---:|---:|
| 0.000–0.500 km | 0.025 km | 21 | 20 |
| 0.550–2.000 km | 0.050 km | 30 | 30 |
| 2.250–10.000 km | 0.250 km | 32 | 32 |
| **Total** | nonuniform | **83** | **82** |

Each measurement cycle consists of one row for each code token `11`, `12`, `13`, and `14`. The eight named scalar values are identical across the four code rows within every detected cycle. Code meaning is not documented in the supplied files.

### Station A microwave file

| Property | Observed value |
|---|---|
| Data rows | 64 |
| Record IDs | Unique and contiguous, 1–64 |
| Cycles | 16 complete four-code cycles |
| Unique timestamps | 16 |
| Timestamp support | 2025-08-02 00:00 through 00:30, every 2 minutes |
| Duplicate timestamp cycles | None |
| Row order | Nondecreasing by timestamp |
| `QCflag` token counts | `9`: 64 |
| Literal coordinate metadata | `118°50'50\"E`, `32°22'07\"N`, opaque token `6.5` |
| Decimal coordinate conversion | 118.8472222222° E, 32.3686111111° N |
| Decoding used | location line: `cp1252`; header: `gb18030`; data tokens: ASCII-compatible |

Observed ranges of the explicitly named scalar columns are listed without treating them as physical-validity tests:

| Header field | Minimum | Maximum | Unique values |
|---|---:|---:|---:|
| `SurTem(℃)` | 25.5 | 25.8 | 4 |
| `SurHum(%)` | 94.1 | 94.5 | 4 |
| `SurPre(hPa)` | 996.8 | 997.3 | 4 |
| `Tir(℃)` | -7.6 | -7.6 | 1 |
| `Rain` | 0 | 0 | 1 |
| `CloudBase(km)` | 6.72 | 6.75 | 4 |
| `Vint(mm)` | 0 | 0 | 1 |
| `Lqint(mm)` | 0 | 0 | 1 |

Opaque-profile coverage and numerical ranges:

| Opaque field | Numeric cells | Minimum | Maximum |
|---|---:|---:|---:|
| `q11` | 1,328 | -27.954 | 25.800 |
| `q12` | 1,328 | 0.004 | 22.654 |
| `q13` | 1,328 | 1.000 | 94.500 |
| `q14` | 1,328 | 0.000 | 2.177 |

### Station B microwave file

| Property | Observed value |
|---|---|
| Data rows | 72 |
| Record IDs | Unique and contiguous, 1–72 |
| Cycles | 18 complete four-code cycles |
| Unique timestamps | 12 |
| Timestamp label support | 2025-08-02 00:02 through 00:30 |
| Duplicate timestamp cycles | 6 timestamp labels have two cycles each |
| Row order | Not nondecreasing by timestamp |
| `QCflag` token counts | `9`: 72 |
| Literal coordinate metadata | `118°35'24\"E`, `32°03'55\"N`, opaque token `9.4` |
| Decimal coordinate conversion | 118.5900000000° E, 32.0652777778° N |
| Decoding used | location and header: UTF-8; data tokens: ASCII-compatible |

The station B header literally contains `SurTem(??)` and `Tir(??)`. The two question marks are preserved; their intended unit text is not reconstructed.

Observed named-column ranges:

| Literal or prefix field | Minimum | Maximum | Unique values |
|---|---:|---:|---:|
| `SurTem(??)` | 25.3 | 25.4 | 2 |
| `SurHum(%)` | 99.1 | 99.2 | 2 |
| `SurPre(hPa)` | 990.9 | 991.3 | 5 |
| `Tir(??)` | 23.7 | 24.7 | 11 |
| `Rain` | 0 | 0 | 1 |
| `CloudBase(km)` | 0.28 | 0.59 | 14 |
| `Vint(mm)` | 64.94 | 80.49 | 17 |
| `Lqint(mm)` | 0.10 | 3.07 | 16 |

Opaque-profile coverage and numerical ranges:

| Opaque field | Numeric cells | Minimum | Maximum |
|---|---:|---:|---:|
| `q11` | 1,494 | -26.619 | 25.400 |
| `q12` | 1,494 | 0.005 | 25.188 |
| `q13` | 1,494 | 1.007 | 99.000 |
| `q14` | 1,494 | 0.000 | 1.758 |

## Wind-profiler files

### Shared schema

All 12 files are ASCII text and have the same structural layout:

- line 1: `WNDROBS 01.20`;
- line 2: six metadata tokens: station ID, longitude, latitude, an opaque numeric value, mode, and timestamp;
- line 3: `ROBS`;
- 62 vertical records;
- final line: `NNNN`.

Each vertical record has seven whitespace-separated tokens: one integer vertical-coordinate token and six undocumented values retained as `W1`–`W6`. Each file has 66 lines, is 2,745 bytes, and ends with a newline. Filename station IDs and timestamps agree with their internal metadata in all 12 files. Mode is `LC` in every file.

The 62-token vertical grid is identical across stations and times, runs from 100 to 11,380, is unique and strictly increasing, and has 12 increments of 60, 10 increments of 120, and 39 increments of 240. The problem statement supports interpreting the first token as height in metres, but the files themselves do not declare its unit or AGL/MSL datum; it is therefore reported as a vertical-coordinate token in unconditional facts.

Slash-only values occur only as whole six-field missing rows. No partially missing row occurs, and no numeric row resumes above the first fully missing row.

### Per-file profiler coverage

| Station | Time | Fully numeric rows | Fully missing rows | Numeric token range | First fully missing token |
|---|---:|---:|---:|---:|---:|
| A | 00:00 | 33 | 29 | 100–4,420 | 4,660 |
| A | 00:06 | 33 | 29 | 100–4,420 | 4,660 |
| A | 00:12 | 33 | 29 | 100–4,420 | 4,660 |
| A | 00:18 | 35 | 27 | 100–4,900 | 5,140 |
| A | 00:24 | 36 | 26 | 100–5,140 | 5,380 |
| A | 00:30 | 37 | 25 | 100–5,380 | 5,620 |
| B | 00:00 | 41 | 21 | 100–6,340 | 6,580 |
| B | 00:06 | 41 | 21 | 100–6,340 | 6,580 |
| B | 00:12 | 41 | 21 | 100–6,340 | 6,580 |
| B | 00:18 | 41 | 21 | 100–6,340 | 6,580 |
| B | 00:24 | 41 | 21 | 100–6,340 | 6,580 |
| B | 00:30 | 41 | 21 | 100–6,340 | 6,580 |

Across the six files, station A has 207 fully numeric and 165 fully missing rows (55.65% and 44.35% of 372 rows); station B has 246 fully numeric and 126 fully missing rows (66.13% and 33.87%). The intersection of fully numeric vertical tokens across all six times is 100–4,420 at A and 100–6,340 at B.

Opaque-field ranges are descriptive only:

| Station | Field | Numeric count | Minimum | Maximum | Unique values |
|---|---|---:|---:|---:|---:|
| A | `W1` | 207 | 39.9 | 323.2 | 186 |
| A | `W2` | 207 | 0.2 | 5.5 | 50 |
| A | `W3` | 207 | -0.3 | 0.3 | 7 |
| A | `W4` | 207 | 100 | 100 | 1 |
| A | `W5` | 207 | 100 | 100 | 1 |
| A | `W6` | 207 | 1.6e-17 | 4.2e-16 | 74 |
| B | `W1` | 246 | 155.4 | 295.6 | 198 |
| B | `W2` | 246 | 4.6 | 8.1 | 35 |
| B | `W3` | 246 | -0.2 | 1.0 | 13 |
| B | `W4` | 246 | 100 | 100 | 1 |
| B | `W5` | 246 | 100 | 100 | 1 |
| B | `W6` | 246 | 2.2e-17 | 9.6e-15 | 126 |

## Structural comparison of station A and station B

| Feature | Station A | Station B |
|---|---|---|
| Sensors supplied | Microwave + profiler | Microwave + profiler |
| Profiler files/times | 6 / same six 6-minute times | 6 / same six 6-minute times |
| Profiler schema/grid | Same as B | Same as A |
| Microwave columns/grid | 95 / 83 heights | 95 / same 83 heights |
| Microwave rows/cycles | 64 / 16 | 72 / 18 |
| Microwave unique times | 16 | 12 |
| Duplicate microwave cycles | None | Six duplicated timestamp labels |
| Microwave chronology | Nondecreasing | Two backward transitions in file order |
| Common all-time numeric profiler top token | 4,420 | 6,340 |
| Microwave/profiler horizontal metadata agreement | About 2.43 m derived separation | About 32.74 km derived separation |
| Microwave header integrity | Unit text displayed | Two unit strings literally degraded to `??` |

## Potential Question 1 use, without selecting a model

The following are candidates for later use only after their semantic and alignment requirements are met:

- For a future combined-instrument reference (`model a`): the explicitly named microwave surface fields, the four opaque microwave vertical arrays, and the six opaque profiler arrays exist on partially overlapping time and vertical supports. Station A's coordinate metadata are horizontally consistent to the precision shown. Station B cannot presently be treated as an instrument pair because its two coordinate records are separated by about 32.74 km.
- For a future profiler-only method (`model b`): `W1`–`W6`, vertical tokens, timestamps, station metadata, and structural missing masks are available. The current files do not establish which `W` field represents any physical wind or turbulence-related variable.
- Slash-filled profiler rows can safely be used as an availability mask, but not as zeros. `QCflag=9` can safely be retained as an observed token, but not accepted or rejected without its code definition.
- `W4` and `W5` are constant at 100 over every numeric record in this subset. That is an observed lack of variation, not evidence about their meaning.

This inventory does not establish a turbulence target, a turbulence formula, or a valid calibration relation between the instruments.
