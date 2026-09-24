
"""
Stage 4.5 targeted preprocessing / interface re-audit.

This script is intentionally conservative. It does not modify data.
It checks the invariants that could otherwise masquerade as a modeling failure.

Because the existing repo already has a validated loader and gap generator in
workspace/experiments/pilot_stage4.py, integrate the adapter functions below
with that loader rather than creating a second interpretation of the files.

Expected outcome:
  PREPROCESSING_REAUDIT_PASS
or
  PREPROCESSING_REAUDIT_BLOCKED

No training is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Any
import numpy as np

ROOT = Path("workspace")
V2 = ROOT / "data" / "processed" / "cleaning_2026e_v2"
RUNS = ROOT / "experiments" / "pilot_runs"
OUT = ROOT / "results" / "pilot" / "stage4_5_preprocessing_reaudit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_unique_ids(samples, split):
    ids = [str(x["id"]) for x in samples]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{split}: duplicate IDs")


def labels_by_id(samples):
    return {
        str(x["id"]): (
            int(x["classification_label"]),
            float(x["regression_label"]),
        )
        for x in samples
    }


def compare_labels(a, b, name_a, name_b):
    if set(a) != set(b):
        raise AssertionError(f"ID mismatch {name_a} vs {name_b}")
    bad = [k for k in a if a[k] != b[k]]
    if bad:
        raise AssertionError(f"label mismatch {name_a} vs {name_b}: {bad[:5]}")


def find_run_config(run_name: str) -> Dict[str, Any]:
    p = RUNS / run_name / "config.json"
    return load_json(p) if p.exists() else {}


def main():
    checks = []
    def ok(name, detail=None):
        checks.append({"name": name, "passed": True, "detail": detail})

    # 1) Frozen v2 / sample identity / labels.
    for version in ["aligned", "unaligned"]:
        for split, expected in [("train", 3395), ("valid", 728)]:
            p = V2 / version / split / "samples.json"
            if not p.exists():
                raise FileNotFoundError(p)
            s = load_json(p)
            assert len(s) == expected, (version, split, len(s))
            assert_unique_ids(s, f"{version}/{split}")
            ok(f"{version}/{split}/count_ids", expected)

    a_tr = load_json(V2 / "aligned" / "train" / "samples.json")
    u_tr = load_json(V2 / "unaligned" / "train" / "samples.json")
    a_va = load_json(V2 / "aligned" / "valid" / "samples.json")
    u_va = load_json(V2 / "unaligned" / "valid" / "samples.json")
    compare_labels(labels_by_id(a_tr), labels_by_id(u_tr), "aligned train", "unaligned train")
    compare_labels(labels_by_id(a_va), labels_by_id(u_va), "aligned valid", "unaligned valid")
    ok("aligned_unaligned_label_identity")

    # 2) Scaler immutability and provenance marker.
    for version in ["aligned", "unaligned"]:
        p = V2 / version / "scalers.json"
        if not p.exists():
            raise FileNotFoundError(p)
        sc = load_json(p)
        blob = json.dumps(sc, sort_keys=True).lower()
        if "train" not in blob:
            raise AssertionError(f"{version}: scalers.json does not visibly record train provenance")
        ok(f"{version}/scaler_sha256", sha256(p))

    # 3) Existing Stage-4 run configs must point to frozen v2 and not A3/A4/test.
    forbidden_tokens = ["attachment3", "attachment4", "a3/", "a4/", "/test", "\\test"]
    for run in ["R0_seed1729", "R0a_seed1729", "R1_seed1729", "R2_seed1729"]:
        cfg = find_run_config(run)
        if not cfg:
            continue
        txt = json.dumps(cfg, sort_keys=True).lower()
        if "cleaning_2026e_v2" not in txt:
            # config formats differ; record rather than invent a failure if path is stored elsewhere.
            checks.append({"name": f"{run}/v2_path_visible", "passed": None,
                           "detail": "config does not expose the data path; verify via existing smoke access log"})
        if any(tok in txt for tok in forbidden_tokens):
            raise AssertionError(f"{run}: forbidden special/test reference in config")
        ok(f"{run}/no_special_test_reference")

    # 4) Mask semantic checks must be executed through the existing loader.
    #    These are hard requirements; fill these adapter calls from pilot_stage4.py:
    adapter_required = [
        "For every injected index: original P==1 and O==1.",
        "After injection: copied P==1 and O==0 only on injected_mask.",
        "Unknown native P/O==2 counts are unchanged outside injected_mask.",
        "R0a and R1 consume the same mask_manifest for same seed/scenario/sample.",
        "Text intervention cache key depends on post-intervention token IDs.",
        "Clean contextual BERT state is never reused for a modified text token sequence.",
        "Scaler tensors are identical across R0a/R1/D0/D1.",
        "Sample/label order is preserved from samples.json -> batch -> predictions.",
    ]
    checks.append({
        "name": "existing_loader_semantic_assertions",
        "passed": None,
        "detail": adapter_required,
    })

    result = {
        "status": "PREPROCESSING_REAUDIT_REQUIRES_LOADER_ASSERTIONS",
        "checks": checks,
        "note": (
            "Do not re-clean data. Add the listed hard assertions to the existing "
            "pilot_stage4.py loader/gap code, run them on train+valid, and only then "
            "upgrade status to PREPROCESSING_REAUDIT_PASS."
        )
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
