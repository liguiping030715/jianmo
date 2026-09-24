"""Bind recovered source maps to the frozen Q1 feature bundle and report scope."""
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


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    manifest = read_csv(HERE / "q1_manifest.csv")
    alignment = read_csv(HERE / "q1_alignment_report.csv")
    ar = {r["id"]: r for r in alignment}
    failures = []
    frame_counts = audio_counts = text_counts = 0
    for row in manifest:
        rid = row["id"]
        p = HERE / "source_maps" / f"{rid}.json"
        f = HERE / "features" / f"{rid}.npz"
        raw = ROOT / "workspace/data/raw" / row["raw_video"]
        if not (p.exists() and f.exists() and raw.exists()):
            failures.append(f"missing file: {rid}")
            continue
        sm = json.loads(p.read_text(encoding="utf-8"))
        prov = json.loads((HERE / "features" / f"{rid}.json").read_text(encoding="utf-8"))
        with np.load(f, allow_pickle=False) as data:
            if len(sm["bins"]) != 50 or sm["id"] != rid:
                failures.append(f"bad bin count or ID: {rid}")
            for b in sm["bins"]:
                k = b["bin"]
                b["P"] = {m: int(data[f"{m}_P"][k]) for m in ("text", "audio", "vision")}
                b["O"] = {m: int(data[f"{m}_O"][k]) for m in ("text", "audio", "vision")}
                if bool(b["video_decoded_frame_ids_by_pts"]) != bool(b["P"]["vision"]):
                    failures.append(f"vision P mismatch: {rid}:{k}")
                if bool(b["audio_lld_frame_ids"]) != bool(b["O"]["audio"]):
                    failures.append(f"audio O mismatch: {rid}:{k}")
                if bool(b["text_word_ids_verified"]) != bool(b["O"]["text"] == 1):
                    failures.append(f"text O mismatch: {rid}:{k}")
            if not np.isfinite(data["text_word_feat"]).all():
                failures.append(f"nonfinite BERT content: {rid}")
            row["text_feature_dim"] = 768
            row["audio_feature_dim"] = 25
            row["vision_feature_dim"] = 1404
            row["common_bins"] = 50
            row["text_word_content_count"] = len(data["text_word_feat"])
            row["text_verified_bins"] = int(np.count_nonzero(data["text_O"] == 1))
            row["text_unknown_bins"] = int(np.count_nonzero(data["text_O"] == 2))
            row["audio_valid_bins"] = int(np.count_nonzero(data["audio_O"] == 1))
            row["vision_valid_bins"] = int(np.count_nonzero(data["vision_O"] == 1))
        sm["feature_npz_sha256"] = sha(f)
        sm["raw_video_sha256"] = sha(raw)
        p.write_text(json.dumps(sm, ensure_ascii=False), encoding="utf-8")
        row["npz_sha256"] = sm["feature_npz_sha256"]
        row["audio_pcm_end_s"] = sm["audio_pcm_end_s"]
        row["video_coverage_end_s"] = prov["video_coverage_end"]
        row["audio_lld_frame_count"] = sm["audio_lld_frame_count"]
        row["source_map_sha256"] = sha(p)
        ar[rid]["audio_pcm_end_s"] = sm["audio_pcm_end_s"]
        ar[rid]["audio_lld_frame_count"] = sm["audio_lld_frame_count"]
        ar[rid]["video_coverage_end_s"] = prov["video_coverage_end"]
        ar[rid]["source_map_sha256"] = sha(p)
        frame_counts += len(sm["video_decoded_frame_pts_s"])
        audio_counts += len(sm["audio_lld_frames"])
        text_counts += len(sm["words"])
    write_csv(HERE / "q1_manifest.csv", manifest)
    write_csv(HERE / "q1_alignment_report.csv", [ar[r["id"]] for r in manifest])

    rid = "-tANM6ETl_M$_$3"
    sm = json.loads((HERE / "source_maps" / f"{rid}.json").read_text(encoding="utf-8"))
    assert sm["duration_s"] == 6.758008
    assert abs(sm["audio_pcm_end_s"] - 6.603625) < 1e-9
    selected = []
    for k in (26, 38, 48, 49):
        b = sm["bins"][k]
        selected.append({"sample_id": rid, "bin": k,
                         "interval_s": f"[{b['start_s']:.6f},{b['end_s']:.6f})",
                         "verified_words": "; ".join(sm["words"][i]["surface"] for i in b["text_word_ids_verified"]),
                         "audio_lld_ids": f"{b['audio_lld_frame_ids'][0]}-{b['audio_lld_frame_ids'][-1]}" if b["audio_lld_frame_ids"] else "none",
                         "video_frame_ids": ";".join(map(str, b["video_decoded_frame_ids_by_pts"])),
                         "video_pts_s": ";".join(f"{v:.6f}" for v in b["video_frame_pts_s"]),
                         "audio_overlap_s": f"{b['audio_timeline_overlap_s']:.6f}",
                         "text_P_O": f"{b['P']['text']}/{b['O']['text']}",
                         "audio_P_O": f"{b['P']['audio']}/{b['O']['audio']}",
                         "vision_P_O": f"{b['P']['vision']}/{b['O']['vision']}"})
    write_csv(HERE / "typical_alignment_evidence.csv", selected)

    expected = {"sample_maps": 100, "bins": 5000, "feature_npz": 100,
                "feature_json": 100, "recovery_log_lines": 100}
    observed = {"sample_maps": len(list((HERE / "source_maps").glob("*.json"))),
                "bins": sum(len(json.loads(p.read_text(encoding="utf-8"))["bins"]) for p in (HERE / "source_maps").glob("*.json")),
                "feature_npz": len(list((HERE / "features").glob("*.npz"))),
                "feature_json": len(list((HERE / "features").glob("*.json"))),
                "recovery_log_lines": len((HERE / "q1_traceability_log.jsonl").read_text(encoding="utf-8").splitlines()),
                "video_frames_mapped": frame_counts, "audio_lld_frames_mapped": audio_counts,
                "bert_words_mapped": text_counts}
    for key, need in expected.items():
        if observed[key] != need:
            failures.append(f"{key}: {observed[key]} != {need}")
    result = {"status": "TRACEABILITY_PARTIAL" if not failures else "TRACEABILITY_AUDIT_FAILED",
              "observed": observed, "failures": failures,
              "limitations": ["The original visual extractor did not persist per-frame FaceMesh success or raw per-frame feature vectors; exact frame contributors to a bin cannot be proven without rerunning it.",
                              "Existing source maps identify exact candidate decoded frame IDs/PTS for every bin and reproduce vision P. Bin O certifies at least one finite visual observation, not which candidate frame contributed.",
                              "No unsupported timestamp is assigned to an unresolved word."]}
    (HERE / "q1_traceability_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
