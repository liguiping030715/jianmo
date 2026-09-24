"""Summarize realized native-index lengths from the saved Q2 factorial manifest."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("workspace/results/q2_evidence")
summary = json.loads((ROOT / "factorial_summary.json").read_text(encoding="utf-8"))
if summary["status"] != "Q2_EVIDENCE_COMPLETE":
    raise RuntimeError("factorial evidence is not complete")
seen = {}
for line in (ROOT / "factorial_mask_manifest.jsonl").open(encoding="utf-8"):
    record = json.loads(line)
    for modality, span in record["spans"].items():
        key = (record["row"], modality, record["requested_ratio"])
        value = (span["native_length"], span["eligible_count"], span["realized_ratio"])
        if key in seen and seen[key] != value:
            raise RuntimeError("factorial manifest length changed across positions/types")
        seen[key] = value
groups = defaultdict(list)
for (row, modality, ratio), (length, eligible, realized) in seen.items():
    groups[(modality, ratio)].append((length, eligible, realized))
rows = []
for (modality, ratio), values in sorted(groups.items()):
    if len(values) != summary["common_cohort_n"]:
        raise RuntimeError("duration summary cohort mismatch")
    lengths = np.asarray([x[0] for x in values])
    realized = np.asarray([x[2] for x in values])
    rows.append({"modality": modality, "requested_ratio": ratio,
                 "samples": len(values), "native_length_median": float(np.median(lengths)),
                 "native_length_p05": float(np.quantile(lengths, .05)),
                 "native_length_p95": float(np.quantile(lengths, .95)),
                 "realized_support_ratio_mean": float(np.mean(realized)),
                 "unit": "native feature/token positions, not seconds"})
with (ROOT / "factorial_duration_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"duration groups: {len(rows)}; cohort per group: {summary['common_cohort_n']}")
