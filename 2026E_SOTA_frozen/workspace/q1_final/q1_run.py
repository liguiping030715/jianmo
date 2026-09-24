"""Q1-FINAL pipeline: process Attachment-1 samples end to end.

Usage:
  python q1_run.py smoke   # deterministic 5 samples, full pipeline
  python q1_run.py all     # all 100 samples
Only Attachment 1 is used. No Attachment 2/3/4 access.
"""
from __future__ import annotations

import sys
import json
import hashlib
import argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q1_media import decode_media
from q1_text import TextModels
from q1_audio import build_smile, audio_features, audio_metadata
from q1_visual import build_face, visual_features, visual_metadata
from q1_align import align_sample

ROOT = HERE.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed" / "cleaning_2026e_v2"
OUT = HERE
FEAT = OUT / "features"
FIG = OUT / "figures"


def load_manifest():
    recs = json.loads((PROC / "attachment1_manifest.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in recs}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def select_smoke(records: list) -> list:
    # Deterministic: shortest, median, longest, plus label diversity.
    rs = sorted(records, key=lambda r: (float(r["container_duration_s"]), r["id"]))
    picks = [rs[0], rs[len(rs) // 2], rs[-1]]
    by_label = {}
    for r in rs:
        by_label.setdefault(r["annotation"], r)
    for lab in ("Negative", "Positive", "Neutral"):
        if lab in by_label and by_label[lab] not in picks:
            picks.append(by_label[lab])
    # de-dup, cap 5
    final = []
    for p in picks:
        if p not in final:
            final.append(p)
    return final[:5]


def process_one(rec, models, smile_pkg, face_pkg):
    smile, opensmile_mod = smile_pkg
    detect, mesh = face_pkg
    mp4 = RAW / rec["raw_video"]
    media = decode_media(mp4)
    text = rec["text"]
    words = models.word_embeddings(text)
    ta = models.forced_align(words, media["pcm16_16k_mono"])
    au = audio_features(smile, media["pcm16_16k_mono"])
    vi = visual_features(detect, mesh, media["video_rgb"], media["video_pts"])
    al = align_sample(media["duration_s"], ta, words, au, vi,
                      media["audio_coverage_end"], media["video_coverage_end"])

    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEAT / f"{rec['id'].replace('/','_')}.npz",
        text_feat=al["text_feat"], text_P=al["text_P"], text_O=al["text_O"],
        audio_feat=al["audio_feat"], audio_P=al["audio_P"], audio_O=al["audio_O"],
        vision_feat=al["vision_feat"], vision_P=al["vision_P"], vision_O=al["vision_O"])

    finite = {m: bool(np.isfinite(al[f"{m}_feat"][al[f"{m}_O"] == 1]).all())
              for m in ("text", "audio", "vision")}
    prov = {
        "id": rec["id"], "mp4": rec["raw_video"],
        "label": rec["label"], "annotation": rec["annotation"],
        "text": text,
        "verified_duration_s": media["duration_s"],
        "old_manifest_duration_s": rec["container_duration_s"],
        "nominal_fps": media["nominal_fps"],
        "n_video_frames_decoded": media["n_video_frames_decoded"],
        "orig_audio_sr": media["orig_audio_sr"],
        "orig_audio_channels": media["orig_audio_channels"],
        "video_coverage_end": media["video_coverage_end"],
        "audio_coverage_end": media["audio_coverage_end"],
        "text_alignment": ta,
        "audio_meta": audio_metadata(opensmile_mod, au),
        "visual_meta": visual_metadata(vi, media["width"], media["height"]),
        "bin_P_counts": {m: int(al[f"{m}_P"].sum()) for m in ("text", "audio", "vision")},
        "bin_O_counts": {m: int(al[f"{m}_O"].sum()) for m in ("text", "audio", "vision")},
        "finite_check": finite,
    }
    (FEAT / f"{rec['id'].replace('/','_')}.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "all"])
    args = ap.parse_args()
    records = load_manifest()
    recs = list(records.values())
    if args.mode == "smoke":
        chosen = select_smoke(recs)
    else:
        chosen = sorted(recs, key=lambda r: r["id"])
    print("samples:", [r["id"] for r in chosen], flush=True)

    models = TextModels()
    smile_pkg = build_smile()
    face_pkg = build_face()
    rows = []
    failures = []
    for i, rec in enumerate(chosen):
        try:
            prov = process_one(rec, models, smile_pkg, face_pkg)
            rows.append(prov)
            print(json.dumps({"i": i, "id": rec["id"],
                              "D": round(prov["verified_duration_s"], 3),
                              "align": prov["text_alignment"]["status"],
                              "O_counts": prov["bin_O_counts"],
                              "finite": prov["finite_check"]}, ensure_ascii=False),
                  flush=True)
        except Exception as exc:
            failures.append({"id": rec["id"], "error": repr(exc)})
            print(json.dumps({"i": i, "id": rec["id"], "FAILED": repr(exc)},
                             ensure_ascii=False), flush=True)
    (OUT / ("smoke_provenance.json" if args.mode == "smoke"
            else "all_provenance.json")).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        (OUT / "run_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        print("FAILURES:", len(failures), flush=True)


if __name__ == "__main__":
    main()
