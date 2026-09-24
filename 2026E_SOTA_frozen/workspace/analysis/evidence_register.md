# Stage 2.5 Evidence Register

## Method and provenance policy

Evidence was collected on 2026-09-16 for metadata recovery only. Searches excluded external competition-solution repositories and were limited to standards, government pages, manufacturer manuals, institutional document hosts, and peer-reviewed technical literature. Supplied raw files and Stage 2 audit artifacts were used to verify structural matches; they were not modified.

Authority labels:

- **A**: official competition documentation, government/industry standard, or official data dictionary.
- **B**: manufacturer documentation or peer-reviewed technical documentation.
- **C**: secondary technical source.
- **D**: inference from names, magnitudes, or common practice.

Only A/B evidence is used to resolve a field. Search failure is recorded rather than replaced by inference.

## External source register

### E-A01 — CAAC document record

- **Source:** Civil Aviation Administration of China, [《民用机场风温廓线雷达系统技术规范》 document page](https://www.caac.gov.cn/XXGK/XXGK/GFXWJ/201511/t20151102_8215.html).
- **Authority:** A.
- **Exact locator/evidence:** page metadata identifies document number `AP-117-TM-2013-01`, issuing office `空管行业管理办公室`, issue date 2013-08-06, and status `有效`; it links the official PDF.
- **Claims supported:** identity, issuing authority, and current official status of the standard.
- **Match limitation:** the landing page alone does not define fields; field claims come from E-A02.

### E-A02 — CAAC AP-117-TM-2013-01 official PDF

- **Source:** CAAC, [official standard PDF](https://www.caac.gov.cn/XXGK/XXGK/GFXWJ/201511/P020151103347220880668.pdf).
- **Authority:** A.
- **Exact locator/evidence:** Appendix 3, especially the filename rules and Tables 14-18 for real-time sampling-height products.
- **Detailed evidence:**
  - Appendix 3 defines the `WNDROBS` / station metadata / `ROBS` / product rows / `NNNN` structure.
  - Table 15 defines metadata order: airport code, longitude, latitude, observation-field elevation above sea level, radar type, and observation time.
  - Table 15 specifies elevation in metres and real-time observation time as observation end time in UTC.
  - Tables 16 and 18 define `ROBS` and `NNNN` as the product start/end markers.
  - Table 17 defines row order and widths: sampling height (5), horizontal wind direction (5), horizontal wind speed (5), vertical wind speed (6), horizontal confidence (3), vertical confidence (3), `Cn²` (8).
  - Table 17 supplies units/sign conventions for wind and confidence fields.
  - Appendix 3 states that missing groups use `/` repeated to the group's nominal width.
  - The filename section defines `Z_RADR_I_iiii_yyyyMMddhhmmss_P_WPRD_<type>_<product>.TXT`, `LC`, and `ROBS`, with filename observation time as end time in UTC.
- **Claims supported:** all resolved profiler row semantics, missing representation, station elevation, product meaning, and profiler time semantics.
- **Exact schema match:** supplied marker sequence and row widths are exact. Example raw widths `5,5,5,6,3,3,8`; missing rows repeat `/` in the same widths.
- **Limitations:** Table 17 does not explicitly give the sampling-height unit or AGL/MSL datum. The supplied `Z_RADA_I_` prefix and five-digit numeric identifier do not match the standard's `Z_RADR_I_` and four-letter airport-code rule.

### E-B01 — Radiometrics TP/WVP-3000 Operator's Manual, January 2006

- **Source:** Radiometrics Corporation manual hosted by NASA GHRC, [TP/WVP-3000 Profiling Radiometer Operator's Manual](https://ghrc.nsstc.nasa.gov/uso/ds_docs/gpmgv/gcpex/gpmradmecgcpex/RADIOMETER_TP_WV3000_UsersManual.pdf).
- **Authority:** B (manufacturer documentation on an official institutional host).
- **Exact locator/evidence:**
  - Section 6.3.2: sequential record number in field 1.
  - Section 6.3.3: timestamp in field 2 and completion/end-time semantics.
  - Section 6.3.4: record type in field 3.
  - Section 6.3.7 / Figure 38: Level 2 code `10` header, `11` temperature, `12` vapor density, `13` relative humidity, `14` liquid water; native profile units K, g/m³, %, g/m³.
  - Section 6.4: timestamp derives from the Windows operating-system clock; UTC synchronization is recommended.
  - Sections 2.3.2 and 5.3.3: rain sensor is a threshold flag for possible radome liquid-water contamination, not rainfall rate or accumulation; native display/file coding is `Y/N`.
  - Sections 2.3.3 and 1.3: IRT measures sky/cloud-base temperature; with retrieved profiles it supports cloud-base altitude.
  - Profile-display and Level 2 descriptions define surface meteorology, integrated vapor, and integrated liquid products.
- **Claims supported:** exact old-software code mapping, scalar physical meanings, end-time semantics, and the non-guarantee of UTC configuration.
- **Exact schema match:** supplied fields 1-3 are `Record`, `DateTime`, and code family `10`-`14`, in the same order. Every supplied cycle contains `11`-`14`.
- **Limitations:** supplied files are tab-separated transformed exports, not the manual's native CSV. The custom 83-height grid, `QCflag`, numeric rain encoding, and converted temperature values are not documented by this manual.

### E-B02 — Radiometrics MP-3000A Operator's Manual, Rev. G

- **Source:** Radiometrics Corporation, [MP-3000A Operator's Manual, Rev. G](https://radiometrics.com/wp-content/uploads/2022/04/MP-3000A-Operator-Manual-RevG.pdf).
- **Authority:** B.
- **Exact locator/evidence:** Section 5.3.7 identifies Level 2 retrievals as temperature, water-vapor density, relative humidity, and liquid-water density; pages around Sections 4-5 identify integrated vapor/liquid and cloud-base products.
- **Claims supported:** corroborates the physical Level 2 product family and scalar meanings.
- **Limitation:** this revision uses record types `100/200/300/400/401-404`, not the supplied `10-14`; it cannot independently map the supplied codes and is used only as corroboration.

## Field-level resolution ledger

| Field/item | Meaning | Unit/convention | Source(s) | Authority | Exact matching reason | Confidence | Final status |
|---|---|---|---|---|---|---|---|
| `ROBS` | Real-time sampling-height product start marker/product type | text | E-A02 | A | Exact marker in every file and exact Appendix 3 framework | High | `RESOLVED` |
| profiler height token | Sampling height | 5-digit integer; unit/datum unknown | E-A02 | A | Exact first position and width | High meaning; none for datum | `PARTIALLY_RESOLVED` |
| `W1` | Horizontal wind direction | degree | E-A02 | A | Exact position 2 and width 5 | High | `RESOLVED` |
| `W2` | Horizontal wind speed | m/s | E-A02 | A | Exact position 3 and width 5 | High | `RESOLVED` |
| `W3` | Vertical wind speed; downward positive, upward negative | m/s | E-A02 | A | Exact position 4, width 6, sign-compatible raw tokens | High | `RESOLVED` |
| `W4` | Horizontal confidence | %, integer 0-100 | E-A02 | A | Exact position 5 and width 3 | High | `RESOLVED` |
| `W5` | Vertical confidence | %, integer 0-100 | E-A02 | A | Exact position 6 and width 3 | High | `RESOLVED` |
| `W6` | Vertical-direction `Cn²` | scientific notation; physical unit not stated | E-A02 | A | Exact position 7, width 8, notation format | High identity; none for unit | Meaning `RESOLVED`; unit `UNRESOLVED` |
| slash tokens | Missing group | `/` repeated to nominal width | E-A02 | A | Exact byte-width pattern in raw rows | High | `RESOLVED` |
| profiler metadata 4 | Observation-field elevation above sea level | m | E-A02 | A | Exact Table 15 position/width and full metadata-row match | High | `RESOLVED` |
| profiler metadata time | End of real-time observation | UTC, `yyyyMMddhhmmss` | E-A02 | A | Exact Table 15 position, width, and format | High | `RESOLVED` |
| filename time | End of observation | UTC | E-A02 | A | Correct time position and exact agreement with internal metadata | Medium-high due prefix/ID deviations | `PARTIALLY_RESOLVED` |
| profiler metadata 1 | Standard would call it four-letter airport code | supplied five-digit numeric token | E-A02 | A | Negative match: width/type conflict | High that it does not conform | `UNRESOLVED` |
| microwave code `11` | Temperature profile | native K; supplied unit unknown | E-B01 | B | Exact field position and complete code-cycle match | High meaning | `PARTIALLY_RESOLVED` |
| microwave code `12` | Vapor-density profile | native g/m³; supplied unit unknown | E-B01 | B | Exact field position and complete code-cycle match | High meaning | `PARTIALLY_RESOLVED` |
| microwave code `13` | Relative-humidity profile | % | E-B01 | B | Exact field position and complete code-cycle match | High | `RESOLVED` |
| microwave code `14` | Liquid-water-density profile | native g/m³; supplied unit unknown | E-B01 | B | Exact field position and complete code-cycle match | High meaning | `PARTIALLY_RESOLVED` |
| `SurTem` | Surface/ambient temperature | °C printed in station A header; station B glyph degraded | E-B01 + supplied header | B + observed fact | Name, position, and Level 2 surface-met context agree | High meaning | `RESOLVED` meaning |
| `SurHum` | Surface RH | % | E-B01 + supplied header | B + observed fact | Name, position, and unit agree | High | `RESOLVED` |
| `SurPre` | Surface pressure | hPa | E-B01 + supplied header | B + observed fact | Name, position, and unit agree | High | `RESOLVED` |
| `Tir` | IRT sky/cloud-base temperature | °C printed in station A header | E-B01 + supplied header | B + observed fact | Exact manufacturer name/function | High meaning | `RESOLVED` meaning |
| `Rain` | Rain-sensor threshold flag / radome wetness warning | numeric export encoding unknown | E-B01 | B | Exact manufacturer scalar field, but encoding differs from native `Y/N` | High meaning; low coding | `PARTIALLY_RESOLVED` |
| `CloudBase` | Cloud-base height | km printed in raw header; datum unknown | E-B01 + supplied header | B + observed fact | Manufacturer defines derived cloud-base altitude | High meaning | `PARTIALLY_RESOLVED` |
| `Vint` | Column-integrated water vapor | mm printed in raw header | E-B01 + supplied header | B + observed fact | Manufacturer defines integrated vapor scalar | High meaning | `RESOLVED` |
| `Lqint` | Column-integrated liquid water | mm printed in raw header | E-B01 + supplied header | B + observed fact | Manufacturer defines integrated liquid scalar | High meaning | `RESOLVED` |
| `QCflag` | Unknown quality-control code | observed `9` only | negative search in E-B01/E-B02 | — | No matching field or code table | High uncertainty | `UNRESOLVED` |
| microwave `DateTime` | Observation-set completion/end time | time zone unknown | E-B01 | B | Exact second-field convention and matching product family | High end-time; none for zone | `PARTIALLY_RESOLVED` |

## Negative evidence and unresolved searches

The following searches did not produce an A/B source that exactly matches the supplied export:

1. No official organizer or manufacturer dictionary was found for the added `QCflag` column or the value `9`.
2. No source tied the transformed profile values in the supplied 83-height export to units for codes `11`, `12`, and `14`. The code-11 conflict with native Kelvin proves at least one conversion occurred.
3. No authoritative source established the microwave file's time zone. The manufacturer describes host-clock behavior and recommends UTC synchronization, which is not evidence that this host was configured in UTC.
4. No exact location-line schema resolved the microwave trailing values `6.5` and `9.4`.
5. No organizer source corrected or explained station B's two coordinate pairs.
6. No source explained station B's distinct duplicate microwave cycles or out-of-order blocks.
7. No official table explicitly stated the unit and AGL/MSL reference of the `ROBS` sampling-height field.

These items remain `UNRESOLVED`; no C/D inference is promoted.

## Supplied-file evidence used for exact-match checks

The following are observed facts already reproduced by Stage 2 and are not external semantic claims:

- all 12 profiler files begin `WNDROBS 01.20`, contain one metadata row, `ROBS`, 62 seven-token product rows, and `NNNN`;
- a complete product row has widths `5,5,5,6,3,3,8`;
- slash-only rows preserve those widths;
- profiler metadata timestamps agree with filename timestamps in every supplied file;
- microwave files contain 95 columns with third-column header `10` and complete `11`-`14` cycles;
- all supplied microwave `QCflag` tokens equal `9`;
- station B duplicate cycles are not byte- or payload-identical;
- the Stage 2 great-circle separations are `2.425342446355008 m` for station A and `32739.65990053063 m` for station B.

Reproducibility sources: `workspace/analysis/audit_artifacts/audit_summary.json` and the Stage 2 scripts under `workspace/analysis/scripts/`.

## AI-assisted research disclosure

Web discovery and document triage were AI-assisted. Each promoted resolution was checked against an official or manufacturer/peer-reviewed source and against the literal supplied file structure. No external solution repository, competition answer, or numerical model result was used.
