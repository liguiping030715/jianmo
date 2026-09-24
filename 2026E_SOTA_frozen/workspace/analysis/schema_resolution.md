# Stage 2.5 Schema Resolution

## Scope and decision rule

This report resolves metadata blockers found in Stage 2 for the supplied Question 1 rehearsal data only. It does not define a turbulence indicator, design model `a` or `b`, align training samples, or alter any raw file.

Statuses are used conservatively:

- `RESOLVED`: Level A or B evidence identifies the field and the supplied structure matches that evidence.
- `PARTIALLY_RESOLVED`: some semantics are supported, but a unit, encoding, datum, time zone, or exact-export detail remains unsupported.
- `UNRESOLVED`: no sufficiently exact Level A/B evidence was found.

Authority levels follow the project rule: A = official standard or organizer/government documentation; B = manufacturer or peer-reviewed technical documentation; C = secondary source; D = inference only. Only A/B evidence changes an unknown field to resolved.

## Blocker register

| ID | Stage 2 blocker | Evidence needed | Stage 2.5 outcome |
|---|---|---|---|
| WP-01 | Meanings/order of profiler `W1`-`W6` | Exact `WNDROBS`/`ROBS` product table | `RESOLVED` by CAAC AP-117-TM-2013-01 Table 17 |
| WP-02 | `ROBS` product meaning | Official product definition | `RESOLVED`: real-time product at sampling heights |
| WP-03 | Slash missing-value convention | Exact product-format rule | `RESOLVED`: repeat `/` to the nominal field width |
| WP-04 | Profiler time semantics and time standard | Exact metadata and filename convention | `RESOLVED`: end of real-time observation, UTC |
| WP-05 | Fourth profiler metadata value | Exact station-metadata table | `RESOLVED`: observation-field elevation above sea level, metres |
| WP-06 | Profiler sampling-height unit and vertical datum | Explicit definition of Table 17 field | `UNRESOLVED`: Table 17 gives integer formatting but not a unit or AGL/MSL datum |
| WP-07 | Filename conformance | Official filename grammar | `PARTIALLY_RESOLVED`: convention found, but supplied names visibly deviate |
| WP-08 | First profiler metadata token | Exact station-code rule matching five numeric characters | `UNRESOLVED`: standard expects a four-letter airport code; supplied values are five-digit numerics |
| MW-01 | Meanings of profile codes `11`-`14` | Manufacturer format with the same code set | `RESOLVED` for physical meanings; current-export units only partly resolved |
| MW-02 | Meanings of named scalar fields | Matching manufacturer definitions | Mostly `RESOLVED`; raw headers independently state several units |
| MW-03 | Meaning of `QCflag=9` | Exact data dictionary/QC table | `UNRESOLVED` |
| MW-04 | Microwave timestamp interval meaning | Matching manufacturer convention | `RESOLVED` as completion/end time for the matched product family |
| MW-05 | Microwave timestamp time zone | Exact export/site configuration | `UNRESOLVED`; manufacturer recommends UTC synchronization but timestamps derive from the host clock |
| MW-06 | Microwave profile-height datum | Exact export/data dictionary | `UNRESOLVED` |
| MW-07 | Station B duplicate-cycle semantics | Organizer/export-system documentation | `UNRESOLVED` |
| ST-01 | Station B coordinate conflict | Organizer correction or station inventory | `UNRESOLVED`; authoritative generic documentation cannot choose between supplied coordinates |
| ST-02 | Microwave location-line values `6.5` and `9.4` | Exact location-line schema | `UNRESOLVED` |

## Wind-profiler `ROBS` resolution

### Why AP-117-TM-2013-01 matches

The official CAAC page identifies AP-117-TM-2013-01 as the effective *Technical Specification for Civil Airport Wind and Temperature Profiler Radar Systems*. Its Appendix 3 defines a text product with the same structural markers as every supplied profiler file: `WNDROBS` plus a five-character version, one station-metadata row, `ROBS`, variable-count product rows, and `NNNN`. It also defines seven fields per product row: sampling height plus six values. This is an exact structural match to the supplied seven-token rows.

The match is also field-width specific. The raw example

`00100 152.4 002.6 -000.2 100 100 2.1E-016`

has token widths `5, 5, 5, 6, 3, 3, 8`, exactly the widths in Table 17. The supplied missing row

`10660 ///// ///// ////// /// /// ////////`

uses the same widths and exactly follows the standard's slash rule. This evidence is stronger than a magnitude-based interpretation.

Primary source: [CAAC document page](https://www.caac.gov.cn/XXGK/XXGK/GFXWJ/201511/t20151102_8215.html) and [official AP-117-TM-2013-01 PDF](https://www.caac.gov.cn/XXGK/XXGK/GFXWJ/201511/P020151103347220880668.pdf).

### Product row fields

| Supplied parser name | Resolved field | Unit/encoding | Status | Authority | Exact evidence and confidence |
|---|---|---|---|---|---|
| vertical token | Sampling height | Five-digit integer; unit and datum unresolved | `PARTIALLY_RESOLVED` | A | Appendix 3 Table 17 field 1 gives the name and width but no unit/datum. High confidence in meaning and formatting; no claim of AGL or MSL. |
| `W1` | Horizontal wind direction | degrees; three integer digits and one decimal | `RESOLVED` | A | Table 17 field 2, width 5. Exact position and width match. High confidence. |
| `W2` | Horizontal wind speed | m/s; three integer digits and one decimal | `RESOLVED` | A | Table 17 field 3, width 5. Exact position and width match. High confidence. |
| `W3` | Vertical wind speed | m/s; signed, three integer digits and one decimal; downward positive, upward negative | `RESOLVED` | A | Table 17 field 4, width 6, including sign convention. Exact match to values such as `-000.2`. High confidence. |
| `W4` | Horizontal confidence | %, integer 0-100 | `RESOLVED` | A | Table 17 field 5, width 3. High confidence. |
| `W5` | Vertical confidence | %, integer 0-100 | `RESOLVED` | A | Table 17 field 6, width 3. High confidence. |
| `W6` | Vertical-direction refractive-index structure constant, `Cn²` | Eight-character scientific notation; physical unit not stated in the CAAC table | Meaning `RESOLVED`; unit `UNRESOLVED` | A | Table 17 field 7 names `Cn²` and gives notation such as `2.6e-024`. High confidence in identity and formatting; no unit is assigned to the supplied export. |

The parser may now expose semantic aliases for `W1`-`W6` in later work, but Stage 2 files and raw files remain unchanged. `Cn²` is a measured/retrieved profiler product field; this resolution does **not** select it as the Question 1 turbulence target.

### Product, missingness, metadata, and time

| Supplied element | Resolution | Unit/convention | Status | Authority |
|---|---|---|---|---|
| `WNDROBS 01.20` | `WNDROBS` file marker plus data-format version `01.20` | Text | `RESOLVED` | A |
| `ROBS` | Start marker for the real-time sampling-height product | Text | `RESOLVED` | A |
| `NNNN` | Product end marker | Text | `RESOLVED` | A |
| Slash-only tokens | Missing group; `/` repeated to nominal field width | Width preserving | `RESOLVED` | A |
| Metadata token 2 | Station longitude | degrees, signed convention in standard | `RESOLVED` | A |
| Metadata token 3 | Station latitude | degrees, signed convention in standard | `RESOLVED` | A |
| Metadata token 4 (`16.2`, `40.6`) | Observation-field elevation above sea level | m | `RESOLVED` | A |
| Metadata token 5 `LC` | L-band boundary-layer wind-profiler radar type code | two-character code | `RESOLVED` | A |
| Metadata token 6 | Real-time observation end time | `yyyyMMddhhmmss`, UTC | `RESOLVED` | A |

The fourth metadata values may therefore be used as site elevations: station A `16.2 m` and station B `40.6 m`. This does not resolve the microwave location-line values `6.5` and `9.4`, nor does it establish the vertical datum of the product row's sampling-height field.

### Filename and identifier caveat

The standard filename grammar is `Z_RADR_I_iiii_yyyyMMddhhmmss_P_WPRD_<radar-type>_<product>.TXT`, where `iiii` is a four-letter airport code; the timestamp is the observation end time in UTC. The supplied names instead begin `Z_RADA_I_`, and their identifier token is the five-digit numeric `58235` or `58238`.

Therefore:

- the timestamp position and `WPRD_LC_ROBS` suffix are structurally consistent with the standard;
- the supplied filename prefix and identifier do not exactly conform;
- `58235` and `58238` remain literal supplied identifiers and must not be relabeled as the standard's four-letter airport code;
- the internal metadata timestamp is the stronger source for the UTC/end-time interpretation because its row otherwise matches Table 15.

### What `ROBS` does not contain

The CAAC standard separately defines radial-data products containing sampling height, spectrum width, signal-to-noise ratio, and radial velocity. The supplied files are `ROBS` processed wind products, not those radial-data products. Consequently, the supplied `ROBS` rows do not contain a documented spectrum-width or radial-velocity field. A later pilot cannot use a spectrum-width or radial-velocity formula unless additional radial-product data are supplied.

## Microwave-radiometer resolution

### Exact-match basis and limitation

The supplied table has `Record` in field 1, `DateTime` in field 2, a third field whose header is `10`, and repeating record types `11`, `12`, `13`, `14`. The Radiometrics TP/WVP-3000 manufacturer manual defines the same first-three-field conventions and exactly this Level 2 code set. This is sufficient Level B evidence for the code meanings.

The current files are nevertheless not untouched manufacturer CSV output: they are tabular exports with a custom 83-height grid, added `QCflag`, and at least one unit conversion. The manufacturer manual defines native code-11 temperature in kelvin, while supplied code-11 surface values are near the supplied Celsius surface-temperature values and the station A header explicitly labels surface temperature in Celsius. Thus the manual's native units cannot be copied mechanically to every supplied profile column.

Manufacturer evidence: [Radiometrics TP/WVP-3000 Operator's Manual, January 2006, Section 6.3](https://ghrc.nsstc.nasa.gov/uso/ds_docs/gpmgv/gcpex/gpmradmecgcpex/RADIOMETER_TP_WV3000_UsersManual.pdf). A newer manufacturer manual confirms the same physical Level 2 products but uses a different record-number scheme, so it is corroborating rather than the exact code dictionary: [Radiometrics MP-3000A Operator's Manual, Rev. G](https://radiometrics.com/wp-content/uploads/2022/04/MP-3000A-Operator-Manual-RevG.pdf).

### Profile codes

| Code | Resolved physical meaning | Manufacturer-native unit | Unit in supplied export | Status | Authority/confidence |
|---|---|---|---|---|---|
| `10` | Header record type for non-GPS Level 2 records; in the supplied file it appears as the third-column header | n/a | n/a | `RESOLVED` | B, high; exact position and code family match |
| `11` | Temperature profile | K | `UNRESOLVED`; supplied values show a conversion relative to the native definition | Meaning `RESOLVED`, unit `UNRESOLVED` | B, high meaning / low unit |
| `12` | Water-vapor density profile | g/m³ | `UNRESOLVED`; no unit is printed over supplied profile columns and the export is transformed | Meaning `RESOLVED`, unit `UNRESOLVED` | B, high meaning / medium-low unit |
| `13` | Relative-humidity profile | % | % is directly tied to the physical quantity in the exact code dictionary | `RESOLVED` | B, high |
| `14` | Liquid-water density profile | g/m³ | `UNRESOLVED`; no unit is printed over supplied profile columns and the export is transformed | Meaning `RESOLVED`, unit `UNRESOLVED` | B, high meaning / medium-low unit |

The Stage 2 magnitude checks remain useful corroboration only; they are not the basis of these resolutions.

### Named scalar fields

| Raw field | Physical meaning | Supplied unit/encoding | Status | Evidence |
|---|---|---|---|---|
| `SurTem` | Surface/ambient air temperature | Station A header: °C; station B's unit glyph is degraded but the schema position matches | `RESOLVED` meaning; station B unit supported by cross-file schema but its literal glyph remains degraded | B manual plus raw header |
| `SurHum` | Surface/ambient relative humidity | % | `RESOLVED` | B manual plus raw header |
| `SurPre` | Surface barometric pressure | hPa | `RESOLVED` | B manual plus raw header |
| `Tir` | Infrared sky/cloud-base temperature measured by the IRT | Station A header: °C; station B glyph degraded | `RESOLVED` meaning; same unit caveat as `SurTem` for station B | B manual plus raw header |
| `Rain` | Rain-sensor threshold flag indicating possible liquid water on the radome; not rain rate or accumulated rainfall | Supplied numeric coding, including `0`, is undocumented | Meaning `RESOLVED`; encoding `UNRESOLVED` | B manual |
| `CloudBase` | Retrieved/derived cloud-base height using IRT and profile information | km in supplied header | `RESOLVED` meaning and printed unit; vertical datum remains unresolved | B manual plus raw header |
| `Vint` | Column-integrated water vapor | mm in supplied header | `RESOLVED` meaning and supplied unit | B manual for meaning; raw header for export unit |
| `Lqint` | Column-integrated liquid water | mm in supplied header | `RESOLVED` meaning and supplied unit | B manual for meaning; raw header for export unit |
| `QCflag` | Unknown | observed token `9`; semantics unknown | `UNRESOLVED` | No exact A/B definition found |

The older manual documents a rain flag as `Y/N`, whereas this export uses numeric tokens. Therefore `Rain=0` is preserved literally; it is not decoded as “no rain” without an export-specific dictionary. Similarly, the manual contains no `QCflag` definition matching this exported column.

### Microwave time semantics

The matched manufacturer manual says the second-field timestamp is the completion/end time of the observation set. It also says timestamps are derived from the Microsoft Windows operating-system clock and recommends synchronization to a UTC reference. A recommendation is not proof of the configuration used for these files.

Accordingly:

- observation-set end/completion semantics: `RESOLVED` at Level B;
- microwave time zone: `UNRESOLVED`;
- equality between microwave labels and profiler UTC labels is still label equality, not proven physical simultaneity.

## Station metadata and cross-station issues

| Item | Observed supplied fact | Stage 2.5 resolution |
|---|---|---|
| Profiler station A coordinates | `118.8472°, 32.3686°` | Longitude/latitude semantics resolved by A |
| Profiler station B coordinates | `118.8994°, 31.9317°` | Longitude/latitude semantics resolved by A |
| Profiler metadata value A/B | `16.2`, `40.6` | Observation-field elevation above sea level, metres, resolved by A |
| Microwave location values | coordinates plus `6.5`/`9.4` | Coordinates readable; final numeric token unresolved |
| Station A inter-instrument separation | Stage 2 derived `2.425 m` | Remains a reproducible derived fact; no new assumption |
| Station B inter-instrument separation | Stage 2 derived `32,739.660 m` | Remains unresolved as a station mismatch; no source authorizes collocation |
| Station B duplicate microwave cycles | six duplicate timestamp labels, distinct cycles | Preserved; acquisition/revision semantics unresolved |

The resolved profiler elevation does not repair the station B coordinate discrepancy and does not authorize using the microwave final location token as an elevation.

## Safely usable variables after Stage 2.5

Without selecting a turbulence formula, the following variable classes are now semantically available for later candidate assessment:

- for a future combined profiler-radiometer model `a`: horizontal wind direction/speed, vertical wind speed, the two confidence fields, `Cn²`, surface temperature/humidity/pressure, IRT temperature, cloud-base height, integrated vapor/liquid, and the physical identities of the four profiles;
- for a future profiler-only model `b`: horizontal wind direction/speed, vertical wind speed, horizontal/vertical confidence, and `Cn²`;
- unavailable from the supplied `ROBS` product: documented Doppler spectrum width, signal-to-noise ratio, and radial velocity.

These are availability statements, not a choice of indicator. Profile codes `11`, `12`, and `14` remain unsafe for unit-dependent calculations in the transformed export, and `QCflag` cannot be used to filter records.

## Unresolved items that remain model-critical

1. The microwave time zone is not established, while profiler times are explicitly UTC.
2. `QCflag=9` has no authoritative meaning, so microwave validity cannot be screened.
3. The supplied-export units for profile codes `11`, `12`, and `14` are not explicitly documented; code `11` demonstrably differs from the manufacturer's native Kelvin convention.
4. The sampling-height unit/datum for profiler rows and the vertical datum for microwave heights remain undocumented.
5. Station B's 32.74 km coordinate disagreement remains unexplained.
6. Station B's repeated, distinct microwave cycles remain unexplained.
7. The supplied filename/identifier convention differs from the official CAAC grammar.

No assumptions have been inserted to close these gaps.
