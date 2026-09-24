"""Recover discarded Q1 timing metadata without rerunning visual extraction.

Reads Attachment 1 audio only for deterministic CTC/openSMILE timing recovery.
Video PTS come from the prior ISO BMFF sample-table audit and are checked
against stored vision P masks. Existing 50-bin feature values are unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import av
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from q1_audio import build_smile, audio_features
from q1_text import TextModels


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def decode_audio_only(path: Path) -> np.ndarray:
    """Replicate q1_media PCM resampling while never decoding video frames."""
    c = av.open(str(path))
    a = c.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for packet in c.demux(a):
        for frame in packet.decode():
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray())
    return np.concatenate(chunks, axis=1)[0].astype(np.int16) if chunks else np.zeros(0, np.int16)


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def main() -> None:
    torch.set_num_threads(4)
    with (HERE / "q1_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    assert len(manifest) == 100
    audit = json.loads((ROOT / "workspace/evidence_recovery/media_timestamp_audit.json").read_text(encoding="utf-8"))
    media = {}
    for row in audit["files"]:
        match = re.search(r"/([^/]+)/([^/]+)\.mp4$", row["path"].replace("\\", "/"))
        if match:
            media[f"{match.group(1)}$_${match.group(2)}"] = row
    assert {r["id"] for r in manifest}.issubset(media)

    wdir = ROOT / "workspace/references/wav2vec2-base-960h"
    freeze = json.loads((HERE / "q1_extractor_freeze.json").read_text(encoding="utf-8"))
    assert sha(wdir / "pytorch_model.bin") == freeze["forced_alignment"]["weights_sha256"]
    aligner = TextModels.__new__(TextModels)
    aligner.w2v = Wav2Vec2ForCTC.from_pretrained(str(wdir), local_files_only=True).eval()
    aligner.w2v_tok = Wav2Vec2CTCTokenizer(str(wdir / "vocab.json"))
    aligner.w2v_feat = Wav2Vec2FeatureExtractor(str(wdir / "preprocessor_config.json"))
    aligner.blank = aligner.w2v_tok.pad_token_id
    smile, _ = build_smile()
    maps = HERE / "source_maps"
    maps.mkdir(exist_ok=True)
    logs = []
    for idx, row in enumerate(manifest):
        rid = row["id"]
        prov = json.loads((HERE / "features" / f"{rid}.json").read_text(encoding="utf-8"))
        raw = ROOT / "workspace/data/raw" / row["raw_video"]
        pcm = decode_audio_only(raw)
        audio_end = len(pcm) / 16000.0
        assert abs(audio_end - prov["audio_coverage_end"]) < 1e-9, rid
        au = audio_features(smile, pcm)
        D = float(row["verified_duration_s"])
        assert abs(D - prov["verified_duration_s"]) < 1e-6
        vtrack = next(t for t in media[rid]["tracks"] if t["kind"] == "vide")
        pts = np.sort(np.asarray(vtrack["presented_pts_seconds"], dtype=np.float64))
        pts -= pts[0]
        assert len(pts) == int(row["n_video_frames_decoded"])
        assert abs(pts[-1] + 1.0 / float(row["nominal_fps"]) - prov["video_coverage_end"]) < 1e-5

        with np.load(HERE / "features" / f"{rid}.npz", allow_pickle=False) as data:
            words = [{"word": str(w), "char_start": int(s), "char_end": int(e), "emb": emb}
                     for w, s, e, emb in zip(data["text_word_surface"],
                                              data["text_word_char_start"],
                                              data["text_word_char_end"],
                                              data["text_word_feat"])]
            result = aligner.forced_align(words, pcm)
            assert result["status"] == row["text_alignment_status"]
            for name in ("n_words", "n_alignable", "n_resolved"):
                assert result[name] == prov["text_alignment"][name], (rid, name)
            state = int(data["text_alignment_state"])
            edges = np.linspace(0.0, D, 51)
            bins = []
            aud_acc = np.zeros((50, au["dim"]), dtype=np.float64)
            aud_w = np.zeros(50, dtype=np.float64)
            txt_acc = np.zeros((50, 768), dtype=np.float64)
            txt_w = np.zeros(50, dtype=np.float64)
            for k in range(50):
                b0, b1 = float(edges[k]), float(edges[k + 1])
                audio_ids = [int(i) for i, (s, e) in enumerate(zip(au["frame_start"], au["frame_end"]))
                             if overlap(float(s), float(e), b0, b1) > 0]
                video_ids = [int(i) for i, t in enumerate(pts)
                             if min(49, int(np.floor(t * 50 / D))) == k]
                word_ids = []
                if state == 1:
                    word_ids = [int(i) for i, w in enumerate(words)
                                if w["align_status"] == "VERIFIED" and
                                overlap(float(w["start_s"]), float(w["end_s"]), b0, b1) > 0]
                for i in audio_ids:
                    weight = overlap(float(au["frame_start"][i]), float(au["frame_end"][i]), b0, b1)
                    if np.isfinite(au["features"][i]).all():
                        aud_acc[k] += weight * au["features"][i]
                        aud_w[k] += weight
                for i in word_ids:
                    w = words[i]
                    weight = overlap(float(w["start_s"]), float(w["end_s"]), b0, b1)
                    txt_acc[k] += weight * w["emb"]
                    txt_w[k] += weight
                assert bool(audio_end > b0 and overlap(0.0, audio_end, b0, b1) > 0) == bool(data["audio_P"][k])
                assert bool(video_ids) == bool(data["vision_P"][k])
                assert bool(audio_ids and aud_w[k] > 0) == bool(data["audio_O"][k])
                assert bool(word_ids) == bool(data["text_O"][k] == 1)
                bins.append({"bin": k, "start_s": b0, "end_s": b1,
                             "text_word_ids_verified": word_ids,
                             "audio_lld_frame_ids": audio_ids,
                             "video_decoded_frame_ids_by_pts": video_ids,
                             "video_frame_pts_s": [float(pts[i]) for i in video_ids],
                             "P": {m: int(data[f"{m}_P"][k]) for m in ("text", "audio", "vision")},
                             "O": {m: int(data[f"{m}_O"][k]) for m in ("text", "audio", "vision")},
                             "audio_timeline_overlap_s": overlap(0.0, audio_end, b0, b1)})
            for k in range(50):
                if data["audio_O"][k] == 1:
                    err = float(np.max(np.abs((aud_acc[k] / aud_w[k]).astype(np.float32) - data["audio_feat"][k])))
                    assert err < 1e-4, (rid, k, "audio", err)
                if data["text_O"][k] == 1:
                    err = float(np.max(np.abs((txt_acc[k] / txt_w[k]).astype(np.float32) - data["text_feat"][k])))
                    assert err < 2e-5, (rid, k, "text", err)
            assert sum(bool(x["audio_lld_frame_ids"]) for x in bins) == int(row["audio_O_bins"])
            assert sum(bool(x["text_word_ids_verified"]) for x in bins) == int(row["text_O_bins"])
        word_report = []
        for i, w in enumerate(words):
            # Sample-level unresolved alignment cannot authorize local text evidence.
            authorized = state == 1 and w["align_status"] == "VERIFIED"
            word_report.append({"word_id": i, "surface": w["word"],
                                "char_start": w["char_start"], "char_end": w["char_end"],
                                "align_status": w["align_status"] if state == 1 else "SAMPLE_UNRESOLVED",
                                "align_conf": w.get("align_conf") if state == 1 else None,
                                "start_s": w.get("start_s") if authorized else None,
                                "end_s": w.get("end_s") if authorized else None})
        source_map = {"id": rid, "raw_video": row["raw_video"], "raw_video_sha256": sha(raw),
                      "feature_npz_sha256": sha(HERE / "features" / f"{rid}.npz"),
                      "duration_s": D, "duration_basis": "PyAV effective edited presentation duration",
                      "raw_mvhd_duration_s": media[rid]["movie_duration_seconds"],
                      "audio_pcm_end_s": audio_end, "audio_lld_frame_count": au["n_frames"],
                      "audio_lld_frames": [{"frame_id": int(i), "start_s": float(s), "end_s": float(e)}
                                           for i, (s, e) in enumerate(zip(au["frame_start"], au["frame_end"]))],
                      "video_decoded_frame_pts_s": [float(x) for x in pts],
                      "video_frame_contributor_limit": "PTS candidate frames exact; per-frame FaceMesh success not persisted in original output. Bin O reports at least one finite frame where O=1.",
                      "text_alignment_state": state, "transcript": prov["text"],
                      "words": word_report, "bins": bins}
        (maps / f"{rid}.json").write_text(json.dumps(source_map, ensure_ascii=False), encoding="utf-8")
        logs.append({"id": rid, "status": "PASS", "duration_s": D,
                     "audio_pcm_end_s": audio_end, "audio_lld_frames": au["n_frames"],
                     "video_frames": len(pts), "text_alignment_status": result["status"],
                     "text_words_resolved": result["n_resolved"], "source_map": str(maps / f"{rid}.json")})
        if idx % 10 == 0:
            print(f"source maps {idx + 1}/100", flush=True)
    (HERE / "q1_traceability_log.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in logs) + "\n", encoding="utf-8")
    assert len(logs) == 100
    print("source maps complete", len(logs), flush=True)


if __name__ == "__main__":
    main()
