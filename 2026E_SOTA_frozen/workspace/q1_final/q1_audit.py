"""Q1 quality audit: build manifest, alignment report and quality report."""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\研究生\jianmo\HuaweiCup-Codex-Final-2026E-SOTA")
Q1 = ROOT / "workspace" / "q1_final"
RAW = ROOT / "workspace" / "data" / "raw"
FEAT = Q1 / "features"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = json.loads((Q1 / "all_provenance.json").read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda r: r["id"])
    manifest, align_rows = [], []
    anomalies = []

    for r in rows:
        rid = r["id"]
        safe = rid.replace("/", "_")
        npz = FEAT / f"{safe}.npz"
        d = np.load(npz)
        po_ok = True
        for m in ("text", "audio", "vision"):
            P, O = d[f"{m}_P"], d[f"{m}_O"]
            if np.any((O == 1) & (P == 0)):
                po_ok = False
        finite = {m: bool(np.isfinite(d[f"{m}_feat"][d[f"{m}_O"] == 1]).all())
                  for m in ("text", "audio", "vision")}
        src = RAW / r["mp4"]
        ta = r["text_alignment"]
        nalign = ta.get("n_alignable", ta.get("n_words"))
        frac = ta.get("n_resolved", 0) / nalign if nalign else 0.0
        npz_sha = sha(npz)

        manifest.append({
            "id": rid, "raw_video": r["mp4"], "source_exists": int(src.exists()),
            "raw_size_bytes": src.stat().st_size if src.exists() else None,
            "verified_duration_s": round(r["verified_duration_s"], 6),
            "old_manifest_duration_s": r["old_manifest_duration_s"],
            "nominal_fps": r["nominal_fps"], "n_video_frames_decoded": r["n_video_frames_decoded"],
            "orig_audio_sr": r["orig_audio_sr"], "orig_audio_channels": r["orig_audio_channels"],
            "label": r["label"], "annotation": r["annotation"],
            "text_alignment_status": ta["status"],
            "n_words": ta.get("n_words"), "n_alignable": nalign,
            "n_words_resolved": ta.get("n_resolved"),
            "text_P_bins": r["bin_P_counts"]["text"], "text_O_bins": r["bin_O_counts"]["text"],
            "audio_P_bins": r["bin_P_counts"]["audio"], "audio_O_bins": r["bin_O_counts"]["audio"],
            "vision_P_bins": r["bin_P_counts"]["vision"], "vision_O_bins": r["bin_O_counts"]["vision"],
            "finite_all": int(all(finite.values())), "po_consistent": int(po_ok),
            "npz_sha256": npz_sha,
        })
        align_rows.append({
            "id": rid, "verified_duration_s": round(r["verified_duration_s"], 6),
            "text_word_resolved_frac": round(frac, 4),
            "text_P_bins": r["bin_P_counts"]["text"], "text_O_bins": r["bin_O_counts"]["text"],
            "audio_P_bins": r["bin_P_counts"]["audio"], "audio_O_bins": r["bin_O_counts"]["audio"],
            "vision_P_bins": r["bin_P_counts"]["vision"], "vision_O_bins": r["bin_O_counts"]["vision"],
            "vision_face_coverage": r["visual_meta"]["face_detection_coverage"],
            "finite_all": int(all(finite.values())), "po_consistent": int(po_ok),
        })
        if not src.exists():
            anomalies.append((rid, "source missing"))
        if not all(finite.values()):
            anomalies.append((rid, "non-finite feature"))
        if not po_ok:
            anomalies.append((rid, "O=1 but P=0"))

    keys = list(manifest[0].keys())
    with open(Q1 / "q1_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)
    with open(Q1 / "q1_alignment_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(align_rows[0].keys()))
        w.writeheader()
        w.writerows(align_rows)

    # quality report
    def col(k):
        return np.array([m[k] for m in manifest], dtype=float)
    n = len(manifest)
    verified = sum(1 for m in manifest if m["text_alignment_status"] == "VERIFIED")
    md = [
        "# Q1 Quality Report", "",
        f"- samples processed: {n}",
        f"- source files present: {int(col('source_exists').sum())}/{n}",
        f"- finite features: {int(col('finite_all').sum())}/{n}",
        f"- P/O consistent (O=1 => P=1): {int(col('po_consistent').sum())}/{n}",
        f"- text forced alignment VERIFIED: {verified}/{n}; UNRESOLVED: {n - verified}/{n}", "",
        "## Verified duration vs old manifest",
        f"- old duration overestimates (rel): mean "
        f"{np.mean([(m['old_manifest_duration_s']-m['verified_duration_s'])/m['old_manifest_duration_s'] for m in manifest]):.3f}",
        "",
        "## Bin coverage (median of O bins / 50)",
    ]
    for m in ("text", "audio", "vision"):
        v = col(f"{m}_O_bins")
        md.append(f"- {m}: median {int(np.median(v))}, mean {v.mean():.1f}, "
                  f"zero-coverage samples {int((v==0).sum())}")
    md += ["", "## Anomalies"]
    md += [f"- {a}: {b}" for a, b in anomalies] or ["- none"]
    unresolved = [m["id"] for m in manifest if m["text_alignment_status"] == "UNRESOLVED"]
    md += ["", "## Text-UNRESOLVED samples"]
    md += [f"- {u}" for u in unresolved]
    (Q1 / "q1_quality_report.md").write_text("\n".join(md), encoding="utf-8")
    print("audit done; samples", n, "verified", verified, "anomalies", len(anomalies))


if __name__ == "__main__":
    main()
