"""Generate the reproducible Stage 2 audit for the supplied Question 1 data.

Run from the repository root:

    python workspace/analysis/scripts/run_stage2_audit.py

The script reads ``workspace/data/raw/station_a`` and ``station_b`` and writes
machine-readable evidence to ``workspace/analysis/audit_artifacts``.  It does
not write to the raw-data tree, drop records, deduplicate cycles, interpolate,
impute, or assign semantics to undocumented fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from question1_parsers import (
    MICROWAVE_NAME,
    PROFILER_SUFFIX,
    MicrowaveFile,
    ProfilerFile,
    haversine_metres,
    microwave_cycles,
    numeric_summary,
    parse_microwave,
    parse_profiler,
    sha256_file,
)


STATIONS = ("station_a", "station_b")
EARTH_RADIUS_M = 6_371_008.8


def iso_minute(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def classify_file(path: Path) -> str:
    if path.name == MICROWAVE_NAME:
        return "microwave_radiometer"
    if path.name.endswith(PROFILER_SUFFIX):
        return "wind_profiler"
    if path.name == ".DS_Store":
        return "filesystem_metadata"
    return "unclassified"


def inventory_files(raw_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for station in STATIONS:
        station_root = raw_root / station
        if not station_root.is_dir():
            raise FileNotFoundError(f"Required station directory not found: {station_root}")
        for path in sorted(item for item in station_root.rglob("*") if item.is_file()):
            kind = classify_file(path)
            records.append(
                {
                    "station": station,
                    "relative_path": rel(path, raw_root),
                    "filename": path.name,
                    "extension": path.suffix,
                    "classification": kind,
                    "is_meteorological_data": kind in {"microwave_radiometer", "wind_profiler"},
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def minute_grid(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    values: list[datetime] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(minutes=step_minutes)
    return values


def height_step_counts(values: Iterable[float]) -> dict[str, int]:
    ordered = list(values)
    counts = Counter(round(right - left, 9) for left, right in zip(ordered, ordered[1:]))
    return {f"{step:g}": count for step, count in sorted(counts.items())}


def summarize_microwave(
    station: str,
    microwave: MicrowaveFile,
    raw_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cycles = microwave_cycles(microwave)
    codes = sorted({row.code for row in microwave.rows})
    timestamps = sorted({row.timestamp for row in microwave.rows})
    timestamp_cycle_counts = Counter(timestamp for timestamp, _ in cycles)
    duplicate_timestamps = {
        iso_minute(timestamp): count
        for timestamp, count in sorted(timestamp_cycle_counts.items())
        if count > 1
    }
    expected_times = minute_grid(timestamps[0], timestamps[-1], 2)
    missing_expected_times = sorted(set(expected_times) - set(timestamps))
    unique_time_steps = sorted(
        {int((right - left).total_seconds() / 60) for left, right in zip(timestamps, timestamps[1:])}
    )
    row_widths = sorted({len(row.raw_fields) for row in microwave.rows})
    record_ids = [row.record_id for row in microwave.rows]
    raw_row_counter = Counter(row.raw_fields for row in microwave.rows)
    payload_counter = Counter(row.raw_fields[1:] for row in microwave.rows)
    named_stats = {
        name: numeric_summary(row.named_values[index] for row in microwave.rows)
        for index, name in enumerate(microwave.named_columns)
    }
    profile_stats = {
        f"q{code}": numeric_summary(
            value
            for row in microwave.rows
            if row.code == code
            for value in row.profile_values
        )
        for code in codes
    }
    qc_counts = Counter(row.qc_token for row in microwave.rows)
    cycle_named_consistency = {
        f"{iso_minute(timestamp)}#{cycle_index}": len(
            {row.named_values for row in by_code.values()}
        )
        == 1
        for (timestamp, cycle_index), by_code in sorted(cycles.items())
    }

    evidence: dict[str, Any] = {}
    for code, prefix in (("11", "SurTem"), ("13", "SurHum")):
        matching_column_indexes = [
            index for index, name in enumerate(microwave.named_columns) if name.startswith(prefix)
        ]
        if not matching_column_indexes:
            evidence[f"q{code}_height0_equals_{prefix}"] = {
                "supported_rows": 0,
                "matching_rows": 0,
                "note": "named column not present",
            }
            continue
        named_index = matching_column_indexes[0]
        relevant = [row for row in microwave.rows if row.code == code]
        absolute_differences = [
            abs(row.profile_values[0] - row.named_values[named_index])
            for row in relevant
        ]
        evidence[f"q{code}_height0_compared_with_{prefix}"] = {
            "supported_rows": len(relevant),
            "matching_rows": sum(
                row.profile_values[0] == row.named_values[named_index] for row in relevant
            ),
            "absolute_difference": numeric_summary(absolute_differences),
            "comparison": "exact numeric equality; semantic meaning remains inferred, not declared",
        }

    cycle_sequence_rows: list[dict[str, Any]] = []
    for file_order_cycle_index, ((timestamp, occurrence), by_code) in enumerate(
        cycles.items(), start=1
    ):
        ordered_rows = sorted(by_code.values(), key=lambda row: row.record_id)
        cycle_sequence_rows.append(
            {
                "station": station,
                "file_order_cycle_index": file_order_cycle_index,
                "timestamp": iso_minute(timestamp),
                "occurrence_at_timestamp": occurrence,
                "record_id_min": min(row.record_id for row in ordered_rows),
                "record_id_max": max(row.record_id for row in ordered_rows),
                "record_ids": ";".join(str(row.record_id) for row in ordered_rows),
                "code_tokens": ";".join(row.code for row in ordered_rows),
                "named_values_consistent_across_codes": len(
                    {row.named_values for row in ordered_rows}
                )
                == 1,
            }
        )
    time_reversals = [
        {
            "from_record_id": left.record_id,
            "from_timestamp": iso_minute(left.timestamp),
            "to_record_id": right.record_id,
            "to_timestamp": iso_minute(right.timestamp),
        }
        for left, right in zip(microwave.rows, microwave.rows[1:])
        if right.timestamp < left.timestamp
    ]

    duplicate_record_rows: list[dict[str, Any]] = []
    duplicate_comparisons: list[dict[str, Any]] = []
    for timestamp, cycle_count in sorted(timestamp_cycle_counts.items()):
        if cycle_count <= 1:
            continue
        for cycle_index in range(1, cycle_count + 1):
            by_code = cycles[(timestamp, cycle_index)]
            for code in codes:
                row = by_code[code]
                row_signature = hashlib.sha256("\t".join(row.raw_fields).encode("utf-8")).hexdigest()
                duplicate_record_rows.append(
                    {
                        "station": station,
                        "timestamp": iso_minute(timestamp),
                        "cycle_index": cycle_index,
                        "code_token": code,
                        "record_id": row.record_id,
                        "source_line": row.source_line,
                        "qc_token": row.qc_token,
                        "named_values_json": json.dumps(row.named_values),
                        "profile_min": min(row.profile_values),
                        "profile_max": max(row.profile_values),
                        "profile_zero_count": sum(value == 0 for value in row.profile_values),
                        "raw_row_sha256": row_signature,
                    }
                )
        baseline = cycles[(timestamp, 1)]
        for cycle_index in range(2, cycle_count + 1):
            comparison = cycles[(timestamp, cycle_index)]
            for code in codes:
                left = baseline[code]
                right = comparison[code]
                named_changes = [
                    name
                    for name, left_value, right_value in zip(
                        microwave.named_columns, left.named_values, right.named_values
                    )
                    if left_value != right_value
                ]
                profile_differences = [
                    abs(left_value - right_value)
                    for left_value, right_value in zip(left.profile_values, right.profile_values)
                ]
                duplicate_comparisons.append(
                    {
                        "station": station,
                        "timestamp": iso_minute(timestamp),
                        "baseline_cycle": 1,
                        "comparison_cycle": cycle_index,
                        "code_token": code,
                        "baseline_record_id": left.record_id,
                        "comparison_record_id": right.record_id,
                        "named_changed_count": len(named_changes),
                        "named_changed_columns": ";".join(named_changes),
                        "profile_changed_count": sum(value != 0 for value in profile_differences),
                        "profile_max_abs_difference": max(profile_differences),
                        "profile_mean_abs_difference": math.fsum(profile_differences)
                        / len(profile_differences),
                        "qc_equal": left.qc_token == right.qc_token,
                        "payload_equal_excluding_record_id": left.raw_fields[1:] == right.raw_fields[1:],
                    }
                )

    summary = {
        "relative_path": rel(microwave.path, raw_root),
        "sha256": sha256_file(microwave.path),
        "bytes": microwave.path.stat().st_size,
        "raw_line_count": microwave.raw_line_count,
        "trailing_newline": microwave.trailing_newline,
        "encoding": {
            "location_line": microwave.location_encoding,
            "header_line": microwave.header_encoding,
            "data_rows": "ASCII-compatible numeric/text tokens",
        },
        "coordinate_metadata": {
            "raw_location_text": microwave.location_text,
            "longitude_deg": microwave.longitude_deg,
            "latitude_deg": microwave.latitude_deg,
            "opaque_location_meta_token": microwave.location_meta_token,
        },
        "schema": {
            "column_count": len(microwave.columns),
            "row_widths": row_widths,
            "columns": list(microwave.columns),
            "named_numeric_columns": list(microwave.named_columns),
            "profile_code_column_header": microwave.columns[2],
            "profile_code_tokens": codes,
            "height_column_count": len(microwave.height_km),
            "height_min_km": min(microwave.height_km),
            "height_max_km": max(microwave.height_km),
            "height_step_km_counts": height_step_counts(microwave.height_km),
        },
        "records": {
            "data_row_count": len(microwave.rows),
            "record_id_min": min(record_ids),
            "record_id_max": max(record_ids),
            "record_ids_unique": len(set(record_ids)) == len(record_ids),
            "record_ids_contiguous_in_file_order": record_ids
            == list(range(record_ids[0], record_ids[0] + len(record_ids))),
            "exact_full_row_duplicate_count": sum(count - 1 for count in raw_row_counter.values() if count > 1),
            "payload_duplicate_count_excluding_record_id": sum(
                count - 1 for count in payload_counter.values() if count > 1
            ),
            "profile_code_counts": dict(sorted(Counter(row.code for row in microwave.rows).items())),
            "qc_token_counts": dict(sorted(qc_counts.items())),
            "blank_field_count": sum(
                field == "" for row in microwave.rows for field in row.raw_fields
            ),
        },
        "time": {
            "first_timestamp": iso_minute(timestamps[0]),
            "last_timestamp": iso_minute(timestamps[-1]),
            "unique_timestamp_count": len(timestamps),
            "cycle_count": len(cycles),
            "unique_time_step_minutes": unique_time_steps,
            "expected_2_minute_grid_count_between_endpoints": len(expected_times),
            "missing_2_minute_grid_times": [iso_minute(value) for value in missing_expected_times],
            "duplicate_timestamp_cycle_counts": duplicate_timestamps,
            "rows_in_nondecreasing_time_order": all(
                left.timestamp <= right.timestamp
                for left, right in zip(microwave.rows, microwave.rows[1:])
            ),
            "time_reversal_transitions": time_reversals,
        },
        "cycle_checks": {
            "codes_per_cycle": codes,
            "all_cycles_complete": True,
            "cycles_with_inconsistent_named_values_across_codes": [
                key for key, is_consistent in cycle_named_consistency.items() if not is_consistent
            ],
        },
        "numeric_summaries": {
            "named_columns": named_stats,
            "opaque_profile_values_by_code": profile_stats,
        },
        "cross_field_equality_evidence": evidence,
    }
    return summary, duplicate_record_rows, duplicate_comparisons, cycle_sequence_rows


def summarize_profiler_file(
    station: str,
    profiler: ProfilerFile,
    raw_root: Path,
) -> dict[str, Any]:
    heights = [row.height_token for row in profiler.rows]
    fully_numeric = [row for row in profiler.rows if row.fully_numeric]
    fully_missing = [row for row in profiler.rows if row.fully_missing]
    partial_missing = [row for row in profiler.rows if any(row.missing_mask) and not row.fully_missing]
    first_missing_index = next(
        (index for index, row in enumerate(profiler.rows) if not row.fully_numeric), None
    )
    numeric_after_first_missing = (
        []
        if first_missing_index is None
        else [
            row.height_token
            for row in profiler.rows[first_missing_index + 1 :]
            if row.fully_numeric
        ]
    )
    return {
        "station": station,
        "relative_path": rel(profiler.path, raw_root),
        "filename": profiler.path.name,
        "sha256": sha256_file(profiler.path),
        "bytes": profiler.path.stat().st_size,
        "raw_line_count": profiler.raw_line_count,
        "trailing_newline": profiler.trailing_newline,
        "format_line": profiler.format_line,
        "station_id": profiler.station_id,
        "filename_station_id": profiler.filename_station_id,
        "station_id_matches_filename": profiler.station_id == profiler.filename_station_id,
        "longitude_deg": profiler.longitude_deg,
        "latitude_deg": profiler.latitude_deg,
        "opaque_metadata_value_4": profiler.metadata_value_4,
        "mode": profiler.mode,
        "timestamp": iso_minute(profiler.timestamp),
        "filename_timestamp": iso_minute(profiler.filename_timestamp),
        "timestamp_matches_filename": profiler.timestamp == profiler.filename_timestamp,
        "data_row_count": len(profiler.rows),
        "row_widths": [7],
        "height_token_min": min(heights),
        "height_token_max": max(heights),
        "height_tokens_unique": len(set(heights)) == len(heights),
        "height_tokens_strictly_increasing": all(
            left < right for left, right in zip(heights, heights[1:])
        ),
        "height_step_counts": height_step_counts(heights),
        "fully_numeric_row_count": len(fully_numeric),
        "fully_missing_row_count": len(fully_missing),
        "partially_missing_row_count": len(partial_missing),
        "numeric_height_min": min((row.height_token for row in fully_numeric), default=None),
        "numeric_height_max": max((row.height_token for row in fully_numeric), default=None),
        "first_non_numeric_height": (
            profiler.rows[first_missing_index].height_token if first_missing_index is not None else None
        ),
        "numeric_heights_after_first_non_numeric": numeric_after_first_missing,
        "missing_token_count_by_opaque_field": {
            f"W{index + 1}": sum(row.missing_mask[index] for row in profiler.rows)
            for index in range(6)
        },
    }


def summarize_profiler_station(
    station: str,
    profilers: list[ProfilerFile],
    file_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(profilers, key=lambda item: item.timestamp)
    timestamps = [item.timestamp for item in ordered]
    grids = [tuple(row.height_token for row in item.rows) for item in ordered]
    station_ids = sorted({item.station_id for item in ordered})
    coordinates = sorted({(item.longitude_deg, item.latitude_deg) for item in ordered})
    metadata_4 = sorted({item.metadata_value_4 for item in ordered})
    modes = sorted({item.mode for item in ordered})
    hashes = Counter(summary["sha256"] for summary in file_summaries)
    exact_duplicate_hash_groups = [digest for digest, count in hashes.items() if count > 1]

    field_stats: dict[str, Any] = {}
    for field_index in range(6):
        field_stats[f"W{field_index + 1}"] = numeric_summary(
            value
            for item in ordered
            for row in item.rows
            for value in [row.values[field_index]]
            if value is not None
        )

    valid_sets = [
        {row.height_token for row in item.rows if row.fully_numeric} for item in ordered
    ]
    common_valid_heights = sorted(set.intersection(*valid_sets)) if valid_sets else []
    return {
        "file_count": len(ordered),
        "station_ids": station_ids,
        "coordinate_pairs": [
            {"longitude_deg": lon, "latitude_deg": lat} for lon, lat in coordinates
        ],
        "opaque_metadata_value_4_values": metadata_4,
        "modes": modes,
        "timestamps": [iso_minute(value) for value in timestamps],
        "unique_time_step_minutes": sorted(
            {int((right - left).total_seconds() / 60) for left, right in zip(timestamps, timestamps[1:])}
        ),
        "timestamps_strictly_increasing": all(
            left < right for left, right in zip(timestamps, timestamps[1:])
        ),
        "all_height_grids_identical": all(grid == grids[0] for grid in grids[1:]),
        "height_grid": list(grids[0]),
        "common_fully_numeric_height_tokens_across_all_times": common_valid_heights,
        "common_fully_numeric_height_min": min(common_valid_heights) if common_valid_heights else None,
        "common_fully_numeric_height_max": max(common_valid_heights) if common_valid_heights else None,
        "exact_duplicate_file_hashes": exact_duplicate_hash_groups,
        "opaque_field_numeric_summaries": field_stats,
        "total_fully_numeric_rows": sum(item["fully_numeric_row_count"] for item in file_summaries),
        "total_fully_missing_rows": sum(item["fully_missing_row_count"] for item in file_summaries),
        "total_partially_missing_rows": sum(item["partially_missing_row_count"] for item in file_summaries),
        "fully_numeric_row_fraction": sum(
            item["fully_numeric_row_count"] for item in file_summaries
        )
        / sum(item["data_row_count"] for item in file_summaries),
        "fully_missing_row_fraction": sum(
            item["fully_missing_row_count"] for item in file_summaries
        )
        / sum(item["data_row_count"] for item in file_summaries),
    }


def build_time_alignment(
    station: str,
    microwave: MicrowaveFile,
    profilers: list[ProfilerFile],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cycles = microwave_cycles(microwave)
    microwave_cycle_counts = Counter(timestamp for timestamp, _ in cycles)
    microwave_times = sorted(microwave_cycle_counts)
    rows: list[dict[str, Any]] = []
    for profiler in sorted(profilers, key=lambda item: item.timestamp):
        deltas = {
            timestamp: abs((timestamp - profiler.timestamp).total_seconds()) / 60.0
            for timestamp in microwave_times
        }
        nearest_delta = min(deltas.values())
        nearest_times = [timestamp for timestamp, delta in deltas.items() if delta == nearest_delta]
        rows.append(
            {
                "station": station,
                "profiler_timestamp": iso_minute(profiler.timestamp),
                "exact_microwave_timestamp_present": profiler.timestamp in microwave_cycle_counts,
                "exact_microwave_cycle_count": microwave_cycle_counts.get(profiler.timestamp, 0),
                "nearest_microwave_timestamps": ";".join(iso_minute(value) for value in nearest_times),
                "nearest_absolute_offset_minutes": nearest_delta,
                "nearest_total_cycle_count": sum(microwave_cycle_counts[value] for value in nearest_times),
            }
        )
    profiler_times = {item.timestamp for item in profilers}
    exact_intersection = sorted(profiler_times & set(microwave_times))
    return rows, {
        "profiler_unique_timestamp_count": len(profiler_times),
        "microwave_unique_timestamp_count": len(microwave_times),
        "exact_common_timestamp_count": len(exact_intersection),
        "exact_common_timestamps": [iso_minute(value) for value in exact_intersection],
        "profiler_timestamps_without_exact_microwave_match": [
            iso_minute(value) for value in sorted(profiler_times - set(microwave_times))
        ],
        "exact_common_cycle_count_counting_duplicate_microwave_cycles": sum(
            microwave_cycle_counts[value] for value in exact_intersection
        ),
        "note": "No matching tolerance or deduplication rule is selected; nearest times are descriptive only.",
    }


def build_height_alignment(
    station: str,
    microwave: MicrowaveFile,
    profilers: list[ProfilerFile],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    microwave_height_m = [round(value * 1000.0, 9) for value in microwave.height_km]
    microwave_height_set = set(microwave_height_m)
    rows: list[dict[str, Any]] = []
    all_file_valid_sets: list[set[int]] = []
    for profiler in sorted(profilers, key=lambda item: item.timestamp):
        valid_heights = [row.height_token for row in profiler.rows if row.fully_numeric]
        all_file_valid_sets.append(set(valid_heights))
        exact = sorted(height for height in valid_heights if height in microwave_height_set)
        nearest_offsets = [
            min(abs(height - microwave_height) for microwave_height in microwave_height_m)
            for height in valid_heights
        ]
        rows.append(
            {
                "station": station,
                "profiler_timestamp": iso_minute(profiler.timestamp),
                "profiler_numeric_height_min_token": min(valid_heights),
                "profiler_numeric_height_max_token": max(valid_heights),
                "numeric_overlap_min_under_metre_correspondence": max(
                    min(valid_heights), min(microwave_height_m)
                ),
                "numeric_overlap_max_under_metre_correspondence": min(
                    max(valid_heights), max(microwave_height_m)
                ),
                "exact_common_height_count_under_metre_correspondence": len(exact),
                "exact_common_height_tokens": ";".join(str(value) for value in exact),
                "nearest_microwave_grid_offset_min": min(nearest_offsets),
                "nearest_microwave_grid_offset_max": max(nearest_offsets),
                "nearest_microwave_grid_offset_mean": math.fsum(nearest_offsets)
                / len(nearest_offsets),
            }
        )
    common_valid = sorted(set.intersection(*all_file_valid_sets))
    exact_common_all_times = sorted(
        height for height in common_valid if height in microwave_height_set
    )
    return rows, {
        "microwave_labeled_height_min_km": min(microwave.height_km),
        "microwave_labeled_height_max_km": max(microwave.height_km),
        "profiler_height_unit_status": (
            "The first profiler field is an opaque height token. The statement supports metre units, "
            "but the file does not declare units or AGL/MSL datum."
        ),
        "alignment_calculation_assumption": (
            "Microwave labeled km values were multiplied by 1000 and compared numerically with "
            "profiler height tokens; this does not resolve vertical datum."
        ),
        "common_profiler_numeric_height_min_across_all_times": min(common_valid),
        "common_profiler_numeric_height_max_across_all_times": max(common_valid),
        "exact_common_height_tokens_across_all_times_under_metre_correspondence": exact_common_all_times,
        "exact_common_height_count_across_all_times_under_metre_correspondence": len(
            exact_common_all_times
        ),
    }


def coordinate_comparison(
    station: str,
    microwave: MicrowaveFile,
    profilers: list[ProfilerFile],
) -> dict[str, Any]:
    coordinates = {(item.longitude_deg, item.latitude_deg) for item in profilers}
    metadata_values = {item.metadata_value_4 for item in profilers}
    if len(coordinates) != 1 or len(metadata_values) != 1:
        raise ValueError(f"Profiler coordinate/metadata values vary within {station}")
    profiler_lon, profiler_lat = next(iter(coordinates))
    profiler_meta = next(iter(metadata_values))
    return {
        "station": station,
        "microwave_longitude_deg": microwave.longitude_deg,
        "microwave_latitude_deg": microwave.latitude_deg,
        "profiler_longitude_deg": profiler_lon,
        "profiler_latitude_deg": profiler_lat,
        "longitude_delta_deg_profiler_minus_microwave": profiler_lon
        - microwave.longitude_deg,
        "latitude_delta_deg_profiler_minus_microwave": profiler_lat
        - microwave.latitude_deg,
        "great_circle_distance_m": haversine_metres(
            microwave.longitude_deg,
            microwave.latitude_deg,
            profiler_lon,
            profiler_lat,
            earth_radius_m=EARTH_RADIUS_M,
        ),
        "great_circle_formula_radius_m": EARTH_RADIUS_M,
        "microwave_opaque_location_meta_token": microwave.location_meta_token,
        "profiler_opaque_metadata_value_4": profiler_meta,
        "opaque_metadata_numeric_difference": profiler_meta
        - microwave.location_meta_token,
        "caution": "The two opaque vertical/site metadata numbers are not assigned units or meaning.",
    }


def main() -> int:
    script_path = Path(__file__).resolve()
    default_workspace = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_workspace / "data" / "raw",
        help="Raw-data root containing station_a and station_b.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_path.parent.parent / "audit_artifacts",
        help="Directory for generated JSON/CSV evidence.",
    )
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise ValueError("Output directory must not be inside the immutable raw-data tree")

    inventory_before = inventory_files(raw_root)
    inventory_by_path = {record["relative_path"]: record for record in inventory_before}
    microwave_files: dict[str, MicrowaveFile] = {}
    profiler_files: dict[str, list[ProfilerFile]] = {}
    for station in STATIONS:
        microwave_path = raw_root / station / MICROWAVE_NAME
        profiler_paths = sorted((raw_root / station / "wind_profiler").glob(f"*{PROFILER_SUFFIX}"))
        if not microwave_path.is_file():
            raise FileNotFoundError(microwave_path)
        if not profiler_paths:
            raise FileNotFoundError(f"No profiler files found under {raw_root / station / 'wind_profiler'}")
        microwave_files[station] = parse_microwave(microwave_path)
        profiler_files[station] = [parse_profiler(path) for path in profiler_paths]

    microwave_summaries: dict[str, Any] = {}
    duplicate_records: list[dict[str, Any]] = []
    duplicate_comparisons: list[dict[str, Any]] = []
    microwave_cycle_sequence: list[dict[str, Any]] = []
    for station in STATIONS:
        summary, records, comparisons, cycle_sequence = summarize_microwave(
            station, microwave_files[station], raw_root
        )
        microwave_summaries[station] = summary
        duplicate_records.extend(records)
        duplicate_comparisons.extend(comparisons)
        microwave_cycle_sequence.extend(cycle_sequence)

    profiler_file_summaries: list[dict[str, Any]] = []
    profiler_station_summaries: dict[str, Any] = {}
    for station in STATIONS:
        station_file_summaries = [
            summarize_profiler_file(station, item, raw_root)
            for item in profiler_files[station]
        ]
        profiler_file_summaries.extend(station_file_summaries)
        profiler_station_summaries[station] = summarize_profiler_station(
            station, profiler_files[station], station_file_summaries
        )

    time_alignment_rows: list[dict[str, Any]] = []
    time_alignment_summary: dict[str, Any] = {}
    height_alignment_rows: list[dict[str, Any]] = []
    height_alignment_summary: dict[str, Any] = {}
    coordinate_summaries: dict[str, Any] = {}
    for station in STATIONS:
        time_rows, time_summary = build_time_alignment(
            station, microwave_files[station], profiler_files[station]
        )
        height_rows, height_summary = build_height_alignment(
            station, microwave_files[station], profiler_files[station]
        )
        time_alignment_rows.extend(time_rows)
        height_alignment_rows.extend(height_rows)
        time_alignment_summary[station] = time_summary
        height_alignment_summary[station] = height_summary
        coordinate_summaries[station] = coordinate_comparison(
            station, microwave_files[station], profiler_files[station]
        )

    inventory_after = inventory_files(raw_root)
    hashes_before = {record["relative_path"]: record["sha256"] for record in inventory_before}
    hashes_after = {record["relative_path"]: record["sha256"] for record in inventory_after}
    if hashes_before != hashes_after:
        raise RuntimeError("Raw-data hashes changed during the audit run")

    summary = {
        "audit_scope": [f"workspace/data/raw/{station}" for station in STATIONS],
        "script": "workspace/analysis/scripts/run_stage2_audit.py",
        "parser_module": "workspace/analysis/scripts/question1_parsers.py",
        "raw_data_immutable_check": {
            "sha256_maps_equal_before_and_after": True,
            "file_count_before": len(inventory_before),
            "file_count_after": len(inventory_after),
        },
        "inventory": {
            "file_count": len(inventory_before),
            "total_bytes": sum(record["bytes"] for record in inventory_before),
            "classification_counts": dict(
                sorted(Counter(record["classification"] for record in inventory_before).items())
            ),
            "meteorological_file_count": sum(
                record["is_meteorological_data"] for record in inventory_before
            ),
        },
        "microwave": microwave_summaries,
        "wind_profiler": {
            "by_station": profiler_station_summaries,
            "file_summaries": profiler_file_summaries,
        },
        "time_alignment": time_alignment_summary,
        "height_alignment": height_alignment_summary,
        "coordinate_comparison": coordinate_summaries,
        "semantic_policy": {
            "microwave_profile_codes": "opaque q11/q12/q13/q14 tokens",
            "profiler_fields": "opaque W1..W6 values",
            "qc": "tokens counted but not interpreted",
            "missing": "slash-only profiler tokens preserved as missing; no imputation",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit_summary.json", summary)
    write_csv(
        output_dir / "file_inventory.csv",
        inventory_before,
        [
            "station",
            "relative_path",
            "filename",
            "extension",
            "classification",
            "is_meteorological_data",
            "bytes",
            "sha256",
        ],
    )
    write_csv(
        output_dir / "profiler_file_summary.csv",
        [
            {
                **row,
                "row_widths": ";".join(map(str, row["row_widths"])),
                "height_step_counts": json.dumps(row["height_step_counts"], sort_keys=True),
                "numeric_heights_after_first_non_numeric": ";".join(
                    map(str, row["numeric_heights_after_first_non_numeric"])
                ),
                "missing_token_count_by_opaque_field": json.dumps(
                    row["missing_token_count_by_opaque_field"], sort_keys=True
                ),
            }
            for row in profiler_file_summaries
        ],
        [
            "station",
            "relative_path",
            "filename",
            "sha256",
            "bytes",
            "raw_line_count",
            "trailing_newline",
            "format_line",
            "station_id",
            "filename_station_id",
            "station_id_matches_filename",
            "longitude_deg",
            "latitude_deg",
            "opaque_metadata_value_4",
            "mode",
            "timestamp",
            "filename_timestamp",
            "timestamp_matches_filename",
            "data_row_count",
            "row_widths",
            "height_token_min",
            "height_token_max",
            "height_tokens_unique",
            "height_tokens_strictly_increasing",
            "height_step_counts",
            "fully_numeric_row_count",
            "fully_missing_row_count",
            "partially_missing_row_count",
            "numeric_height_min",
            "numeric_height_max",
            "first_non_numeric_height",
            "numeric_heights_after_first_non_numeric",
            "missing_token_count_by_opaque_field",
        ],
    )
    write_csv(
        output_dir / "microwave_cycle_sequence.csv",
        microwave_cycle_sequence,
        [
            "station",
            "file_order_cycle_index",
            "timestamp",
            "occurrence_at_timestamp",
            "record_id_min",
            "record_id_max",
            "record_ids",
            "code_tokens",
            "named_values_consistent_across_codes",
        ],
    )
    write_csv(
        output_dir / "microwave_duplicate_records.csv",
        duplicate_records,
        [
            "station",
            "timestamp",
            "cycle_index",
            "code_token",
            "record_id",
            "source_line",
            "qc_token",
            "named_values_json",
            "profile_min",
            "profile_max",
            "profile_zero_count",
            "raw_row_sha256",
        ],
    )
    write_csv(
        output_dir / "microwave_duplicate_comparisons.csv",
        duplicate_comparisons,
        [
            "station",
            "timestamp",
            "baseline_cycle",
            "comparison_cycle",
            "code_token",
            "baseline_record_id",
            "comparison_record_id",
            "named_changed_count",
            "named_changed_columns",
            "profile_changed_count",
            "profile_max_abs_difference",
            "profile_mean_abs_difference",
            "qc_equal",
            "payload_equal_excluding_record_id",
        ],
    )
    write_csv(
        output_dir / "time_alignment.csv",
        time_alignment_rows,
        [
            "station",
            "profiler_timestamp",
            "exact_microwave_timestamp_present",
            "exact_microwave_cycle_count",
            "nearest_microwave_timestamps",
            "nearest_absolute_offset_minutes",
            "nearest_total_cycle_count",
        ],
    )
    write_csv(
        output_dir / "height_alignment.csv",
        height_alignment_rows,
        [
            "station",
            "profiler_timestamp",
            "profiler_numeric_height_min_token",
            "profiler_numeric_height_max_token",
            "numeric_overlap_min_under_metre_correspondence",
            "numeric_overlap_max_under_metre_correspondence",
            "exact_common_height_count_under_metre_correspondence",
            "exact_common_height_tokens",
            "nearest_microwave_grid_offset_min",
            "nearest_microwave_grid_offset_max",
            "nearest_microwave_grid_offset_mean",
        ],
    )
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "files_written": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "inventory_files": len(inventory_by_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    raise SystemExit(main())
