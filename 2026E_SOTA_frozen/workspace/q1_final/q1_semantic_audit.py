"""Lightweight post-patch audit of all Q1 feature records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    official = {r["id"]: r for r in json.loads((ROOT / "workspace/data/processed/cleaning_2026e_v2/attachment1_manifest.json").read_text(encoding="utf-8"))}
    manifest = load_csv(HERE / "q1_manifest.csv")
    alignment = load_csv(HERE / "q1_alignment_report.csv")
    provenance = json.loads((HERE / "all_provenance.json").read_text(encoding="utf-8"))
    hashes = json.loads((HERE / "q1_semantic_patch_source_hashes.json").read_text(encoding="utf-8"))
    failures = []
    counts = {"records": 0, "transcript_preserved": 0, "bert_content_preserved": 0,
              "verified": 0, "unresolved": 0, "unknown_not_absent": 0,
              "finite_observed": 0, "finite_model_input": 0,
              "label_preserved": 0, "extractor_hashes_unchanged": 0}
    if not (len(official) == len(manifest) == len(alignment) == len(provenance) == 100):
        failures.append("record counts or ID sets differ from 100")
    if not ({r["id"] for r in manifest} == {r["id"] for r in alignment}
            == {r["id"] for r in provenance} == set(official)):
        failures.append("ID sets differ")
    ar_by_id = {r["id"]: r for r in alignment}
    pr_by_id = {r["id"]: r for r in provenance}
    for row in manifest:
        rid = row["id"]
        try:
            rec = official[rid]
            meta = json.loads((HERE / "features" / f"{rid}.json").read_text(encoding="utf-8"))
            with np.load(HERE / "features" / f"{rid}.npz", allow_pickle=False) as data:
                counts["records"] += 1
                if rec["text"] == meta["text"] == pr_by_id[rid]["text"] and rec["text"]:
                    counts["transcript_preserved"] += 1
                if rec["label"] == meta["label"] and rec["annotation"] == meta["annotation"]:
                    counts["label_preserved"] += 1
                state = int(data["text_alignment_state"])
                status = meta["text_alignment"]["status"]
                assert state == int(row["text_alignment_state"]) == int(ar_by_id[rid]["text_alignment_state"])
                assert status == row["text_alignment_status"] == ar_by_id[rid]["text_alignment_status"]
                assert int(data["text_content_present"]) == 1
                words = data["text_word_feat"]
                spans = list(zip(data["text_word_char_start"], data["text_word_char_end"], data["text_word_surface"]))
                assert words.shape == (int(row["n_words"]), 768) and len(spans) == len(words)
                assert all(rec["text"][int(s):int(e)] == str(w) for s, e, w in spans)
                assert np.isfinite(words).all() and np.isfinite(data["text_utterance_feat"]).all()
                counts["bert_content_preserved"] += 1
                P, O = data["text_P"], data["text_O"]
                if status == "VERIFIED":
                    assert state == 1 and set(np.unique(P)).issubset({1, 2}) and set(np.unique(O)).issubset({1, 2})
                    counts["verified"] += 1
                elif status == "UNRESOLVED":
                    assert state == 2 and np.all(P == 2) and np.all(O == 2)
                    assert int(row["text_P_unknown_bins"]) == int(row["text_O_unknown_bins"]) == 50
                    counts["unresolved"] += 1
                    counts["unknown_not_absent"] += 1
                else:
                    raise AssertionError("unexpected alignment status")
                valid_finite = safe_finite = True
                for mod in ("text", "audio", "vision"):
                    obs = data[f"{mod}_O"]
                    pad = data[f"{mod}_P"]
                    valid_finite &= bool(np.isfinite(data[f"{mod}_feat"][obs == 1]).all())
                    safe = data[f"{mod}_model_input"]
                    safe_finite &= bool(np.isfinite(safe).all())
                    assert not np.any((obs == 1) & (pad != 1))
                    assert np.array_equal(safe[obs == 1], data[f"{mod}_feat"][obs == 1])
                    assert np.all(safe[obs != 1] == 0)
                counts["finite_observed"] += int(valid_finite)
                counts["finite_model_input"] += int(safe_finite)
                assert sha(HERE / "features" / f"{rid}.npz") == row["npz_sha256"]
        except Exception as exc:
            failures.append(f"{rid}: {exc!r}")
    for name, expected in hashes.items():
        if sha(ROOT / name) == expected:
            counts["extractor_hashes_unchanged"] += 1
        else:
            failures.append(f"frozen source/model hash changed: {name}")
    assert counts["extractor_hashes_unchanged"] == len(hashes)
    required = {"records": 100, "transcript_preserved": 100, "bert_content_preserved": 100,
                "verified": 78, "unresolved": 22, "unknown_not_absent": 22,
                "finite_observed": 100, "finite_model_input": 100,
                "label_preserved": 100}
    for k, expected in required.items():
        if counts[k] != expected:
            failures.append(f"{k}: {counts[k]} != {expected}")
    result = {"status": "Q1_COMPLETE" if not failures else "Q1_PATCH_BLOCKED",
              "counts": counts, "frozen_hash_count": len(hashes), "failures": failures,
              "note": "Original nonobserved *_feat entries may be NaN. All observed entries and *_model_input arrays are finite."}
    (HERE / "q1_semantic_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
