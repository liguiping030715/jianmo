"""Read-only parsers for the supplied Question 1 rehearsal files.

The module deliberately preserves undocumented fields as opaque tokens:

* microwave profile codes are ``q11`` through ``q14``;
* the six values after profiler height are ``W1`` through ``W6``;
* quality/status tokens are retained but never interpreted.

Only Python's standard library is required.  Parsing fails loudly when a
record width, numeric token, timestamp, or required marker is malformed.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


MICROWAVE_NAME = "microwave_radiometer.txt"
PROFILER_SUFFIX = "_P_WPRD_LC_ROBS.TXT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_strict(data: bytes, encodings: Iterable[str]) -> tuple[str, str]:
    errors: list[str] = []
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError("No declared encoding decoded the bytes: " + " | ".join(errors))


def _finite_float(token: str, *, context: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(f"Expected numeric token at {context}, got {token!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric token at {context}: {token!r}")
    return value


def _dms_to_decimal(degrees: float, minutes: float, seconds: float, hemi: str) -> float:
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if hemi.upper() in {"W", "S"}:
        value = -value
    return value


LOCATION_RE = re.compile(
    r"^\s*"
    r"(?P<lon_d>\d{1,3})°(?P<lon_m>\d{1,2})'(?P<lon_s>\d+(?:\.\d+)?)\"(?P<lon_h>[EW])"
    r"[\t, ]+"
    r"(?P<lat_d>\d{1,2})°(?P<lat_m>\d{1,2})'(?P<lat_s>\d+(?:\.\d+)?)\"(?P<lat_h>[NS])"
    r"[\t, ]+(?P<meta>[-+]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class MicrowaveRow:
    source_line: int
    record_id: int
    timestamp: datetime
    code: str
    named_values: tuple[float, ...]
    profile_values: tuple[float, ...]
    qc_token: str
    raw_fields: tuple[str, ...]


@dataclass(frozen=True)
class MicrowaveFile:
    path: Path
    location_text: str
    location_encoding: str
    header_encoding: str
    longitude_deg: float
    latitude_deg: float
    location_meta_token: float
    columns: tuple[str, ...]
    named_columns: tuple[str, ...]
    height_km: tuple[float, ...]
    rows: tuple[MicrowaveRow, ...]
    raw_line_count: int
    trailing_newline: bool


def parse_microwave(path: Path) -> MicrowaveFile:
    raw = path.read_bytes()
    raw_lines = raw.splitlines()
    if len(raw_lines) < 3:
        raise ValueError(f"Microwave file has fewer than 3 lines: {path}")

    location_text, location_encoding = _decode_strict(
        raw_lines[0], ("utf-8", "cp1252", "gb18030")
    )
    header_text, header_encoding = _decode_strict(
        raw_lines[1], ("utf-8", "gb18030", "cp1252")
    )
    match = LOCATION_RE.match(location_text.rstrip("\t "))
    if not match:
        raise ValueError(f"Cannot parse microwave location line in {path}: {location_text!r}")
    longitude = _dms_to_decimal(
        float(match["lon_d"]), float(match["lon_m"]), float(match["lon_s"]), match["lon_h"]
    )
    latitude = _dms_to_decimal(
        float(match["lat_d"]), float(match["lat_m"]), float(match["lat_s"]), match["lat_h"]
    )

    columns = tuple(header_text.split("\t"))
    height_indices: list[int] = []
    height_km: list[float] = []
    for index, name in enumerate(columns):
        height_match = re.fullmatch(r"(\d+(?:\.\d+)?)\(km\)", name)
        if height_match:
            height_indices.append(index)
            height_km.append(float(height_match.group(1)))
    if not height_indices:
        raise ValueError(f"No labeled height columns in {path}")
    if height_indices != list(range(min(height_indices), max(height_indices) + 1)):
        raise ValueError(f"Microwave height columns are not contiguous in {path}")
    if columns[0:3] != ("Record", "DateTime", "10"):
        raise ValueError(f"Unexpected leading microwave columns in {path}: {columns[0:3]!r}")
    if columns[-1] != "QCflag":
        raise ValueError(f"Unexpected final microwave column in {path}: {columns[-1]!r}")

    profile_start = min(height_indices)
    profile_end = max(height_indices) + 1
    named_columns = columns[3:profile_start]
    rows: list[MicrowaveRow] = []
    for source_line, raw_line in enumerate(raw_lines[2:], start=3):
        if not raw_line.strip():
            continue
        line_text, _ = _decode_strict(raw_line, ("ascii", "utf-8", "gb18030", "cp1252"))
        fields = tuple(line_text.split("\t"))
        if len(fields) != len(columns):
            raise ValueError(
                f"Microwave row width mismatch in {path} line {source_line}: "
                f"expected {len(columns)}, got {len(fields)}"
            )
        try:
            record_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"Invalid record id in {path} line {source_line}: {fields[0]!r}") from exc
        try:
            timestamp = datetime.strptime(fields[1], "%Y/%m/%d %H:%M")
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp in {path} line {source_line}: {fields[1]!r}") from exc
        code = fields[2]
        named_values = tuple(
            _finite_float(value, context=f"{path} line {source_line} column {columns[index]}")
            for index, value in enumerate(fields[3:profile_start], start=3)
        )
        profile_values = tuple(
            _finite_float(value, context=f"{path} line {source_line} column {columns[index]}")
            for index, value in enumerate(fields[profile_start:profile_end], start=profile_start)
        )
        rows.append(
            MicrowaveRow(
                source_line=source_line,
                record_id=record_id,
                timestamp=timestamp,
                code=code,
                named_values=named_values,
                profile_values=profile_values,
                qc_token=fields[-1],
                raw_fields=fields,
            )
        )
    if not rows:
        raise ValueError(f"Microwave file has no data records: {path}")
    return MicrowaveFile(
        path=path,
        location_text=location_text.rstrip("\t "),
        location_encoding=location_encoding,
        header_encoding=header_encoding,
        longitude_deg=longitude,
        latitude_deg=latitude,
        location_meta_token=float(match["meta"]),
        columns=columns,
        named_columns=named_columns,
        height_km=tuple(height_km),
        rows=tuple(rows),
        raw_line_count=len(raw_lines),
        trailing_newline=raw.endswith((b"\n", b"\r")),
    )


@dataclass(frozen=True)
class ProfilerRow:
    source_line: int
    height_token: int
    raw_values: tuple[str, ...]
    values: tuple[float | None, ...]
    missing_mask: tuple[bool, ...]

    @property
    def fully_numeric(self) -> bool:
        return not any(self.missing_mask)

    @property
    def fully_missing(self) -> bool:
        return all(self.missing_mask)


@dataclass(frozen=True)
class ProfilerFile:
    path: Path
    format_line: str
    station_id: str
    longitude_deg: float
    latitude_deg: float
    metadata_value_4: float
    mode: str
    timestamp: datetime
    filename_station_id: str
    filename_timestamp: datetime
    rows: tuple[ProfilerRow, ...]
    raw_line_count: int
    trailing_newline: bool


PROFILER_FILENAME_RE = re.compile(
    r"^Z_RADA_I_(?P<station>\d+)_(?P<time>\d{14})_P_WPRD_LC_ROBS\.TXT$"
)


def _slash_missing(token: str) -> bool:
    return bool(token) and set(token) == {"/"}


def parse_profiler(path: Path) -> ProfilerFile:
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UnicodeError(f"Profiler file is not ASCII: {path}") from exc
    lines = text.splitlines()
    if len(lines) < 5:
        raise ValueError(f"Profiler file too short: {path}")
    if lines[0] != "WNDROBS 01.20":
        raise ValueError(f"Unexpected profiler format line in {path}: {lines[0]!r}")
    metadata = lines[1].split()
    if len(metadata) != 6:
        raise ValueError(f"Expected 6 profiler metadata tokens in {path}, got {len(metadata)}")
    if lines[2] != "ROBS" or lines[-1] != "NNNN":
        raise ValueError(f"Missing ROBS/NNNN marker in {path}")

    filename_match = PROFILER_FILENAME_RE.match(path.name)
    if not filename_match:
        raise ValueError(f"Unexpected profiler filename: {path.name}")
    try:
        timestamp = datetime.strptime(metadata[5], "%Y%m%d%H%M%S")
        filename_timestamp = datetime.strptime(filename_match["time"], "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(f"Invalid profiler timestamp in {path}") from exc

    rows: list[ProfilerRow] = []
    for source_line, line in enumerate(lines[3:-1], start=4):
        fields = tuple(line.split())
        if len(fields) != 7:
            raise ValueError(
                f"Profiler row width mismatch in {path} line {source_line}: "
                f"expected 7, got {len(fields)}"
            )
        try:
            height = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"Invalid profiler height token in {path} line {source_line}") from exc
        raw_values = fields[1:]
        missing_mask = tuple(_slash_missing(value) for value in raw_values)
        values: list[float | None] = []
        for field_index, (value, is_missing) in enumerate(zip(raw_values, missing_mask), start=1):
            values.append(
                None
                if is_missing
                else _finite_float(value, context=f"{path} line {source_line} W{field_index}")
            )
        rows.append(
            ProfilerRow(
                source_line=source_line,
                height_token=height,
                raw_values=raw_values,
                values=tuple(values),
                missing_mask=missing_mask,
            )
        )
    if not rows:
        raise ValueError(f"Profiler file has no data rows: {path}")
    return ProfilerFile(
        path=path,
        format_line=lines[0],
        station_id=metadata[0],
        longitude_deg=_finite_float(metadata[1], context=f"{path} metadata longitude"),
        latitude_deg=_finite_float(metadata[2], context=f"{path} metadata latitude"),
        metadata_value_4=_finite_float(metadata[3], context=f"{path} metadata value 4"),
        mode=metadata[4],
        timestamp=timestamp,
        filename_station_id=filename_match["station"],
        filename_timestamp=filename_timestamp,
        rows=tuple(rows),
        raw_line_count=len(lines),
        trailing_newline=raw.endswith((b"\n", b"\r")),
    )


def microwave_cycles(microwave: MicrowaveFile) -> dict[tuple[datetime, int], dict[str, MicrowaveRow]]:
    """Return cycles without collapsing duplicate timestamps.

    Occurrence 1 is the first row for each code at a timestamp, occurrence 2
    the second, and so on.  A complete cycle must contain the same code set as
    the whole file; otherwise parsing stops because cycle assignment is unsafe.
    """

    expected_codes = set(row.code for row in microwave.rows)
    occurrences: Counter[tuple[datetime, str]] = Counter()
    cycles: dict[tuple[datetime, int], dict[str, MicrowaveRow]] = defaultdict(dict)
    for row in microwave.rows:
        key = (row.timestamp, row.code)
        occurrences[key] += 1
        cycle_key = (row.timestamp, occurrences[key])
        if row.code in cycles[cycle_key]:
            raise ValueError(f"Internal cycle collision in {microwave.path}: {cycle_key} {row.code}")
        cycles[cycle_key][row.code] = row
    incomplete = {
        key: sorted(expected_codes - set(by_code))
        for key, by_code in cycles.items()
        if set(by_code) != expected_codes
    }
    if incomplete:
        raise ValueError(f"Incomplete microwave cycles in {microwave.path}: {incomplete}")
    return dict(cycles)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = list(values)
    if not collected:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "zero_count": 0,
            "negative_count": 0,
            "unique_count": 0,
        }
    return {
        "count": len(collected),
        "min": min(collected),
        "max": max(collected),
        "mean": math.fsum(collected) / len(collected),
        "zero_count": sum(value == 0 for value in collected),
        "negative_count": sum(value < 0 for value in collected),
        "unique_count": len(set(collected)),
    }


def haversine_metres(
    lon1_deg: float,
    lat1_deg: float,
    lon2_deg: float,
    lat2_deg: float,
    *,
    earth_radius_m: float = 6_371_008.8,
) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1_deg, lat1_deg, lon2_deg, lat2_deg))
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 2 * earth_radius_m * math.asin(math.sqrt(a))
