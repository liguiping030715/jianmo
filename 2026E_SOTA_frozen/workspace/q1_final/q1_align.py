"""Q1 common temporal alignment to K=50 bins.

Text uses forced-aligned word intervals with overlap-weighted aggregation
(never token index). Audio/vision aggregate by measured timestamps. P and O
are stored separately; missingness is never inferred from zero vectors.
"""
from __future__ import annotations

import numpy as np

K = 50


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_sample(D: float, text_align: dict, words: list, audio: dict,
                 visual: dict, audio_end: float, video_end: float) -> dict:
    edges = np.array([k * D / K for k in range(K + 1)])

    def bins_of(s, e):
        return range(min(K - 1, int(np.floor(s * K / D))),
                     min(K, int(np.floor(e * K / D)) + 1))

    # ---- TEXT ----
    t_feat = np.full((K, 768), np.nan, dtype=np.float32)
    tP = np.zeros(K, dtype=np.uint8)
    tO = np.zeros(K, dtype=np.uint8)
    if text_align["status"] == "VERIFIED":
        acc = np.zeros((K, 768), dtype=np.float64)
        wsum = np.zeros(K, dtype=np.float64)
        for w in words:
            if w["align_status"] != "VERIFIED":
                continue
            s, e = w["start_s"], w["end_s"]
            for b in bins_of(s, e):
                ov = overlap(s, e, edges[b], edges[b + 1])
                if ov > 0:
                    acc[b] += ov * w["emb"]
                    wsum[b] += ov
                    tP[b] = 1
        for b in range(K):
            if wsum[b] > 0:
                t_feat[b] = (acc[b] / wsum[b]).astype(np.float32)
                tO[b] = 1

    # ---- AUDIO (overlap-weighted mean of LLD frames) ----
    Nd = audio["dim"]
    a_acc = np.zeros((K, Nd), dtype=np.float64)
    a_w = np.zeros(K, dtype=np.float64)
    aP = np.zeros(K, dtype=np.uint8)
    aO = np.zeros(K, dtype=np.uint8)
    # P: bins intersecting the actual audio timeline [0, audio_end]
    for b in bins_of(0.0, audio_end):
        if overlap(0.0, audio_end, edges[b], edges[b + 1]) > 0:
            aP[b] = 1
    fs, fe = audio["frame_start"], audio["frame_end"]
    for i in range(audio["n_frames"]):
        for b in bins_of(fs[i], fe[i]):
            ov = overlap(fs[i], fe[i], edges[b], edges[b + 1])
            if ov > 0 and np.isfinite(audio["features"][i]).all():
                a_acc[b] += ov * audio["features"][i]
                a_w[b] += ov
    a_feat = np.full((K, Nd), np.nan, dtype=np.float32)
    for b in range(K):
        if a_w[b] > 0:
            a_feat[b] = (a_acc[b] / a_w[b]).astype(np.float32)
            aO[b] = 1

    # ---- VISION (frames by PTS; P=decoded frame present, O=face observed) ----
    Vd = visual["dim"]
    v_acc = np.zeros((K, Vd), dtype=np.float64)
    v_w = np.zeros(K, dtype=np.float64)
    vP = np.zeros(K, dtype=np.uint8)
    vO = np.zeros(K, dtype=np.uint8)
    for i in range(visual["n_frames"]):
        b = min(K - 1, int(np.floor(visual["pts"][i] * K / D)))
        vP[b] = 1  # a decoded frame exists regardless of face
        if visual["status"][i] == 1 and np.isfinite(visual["features"][i]).all():
            v_acc[b] += visual["features"][i]
            v_w[b] += 1
    v_feat = np.full((K, Vd), np.nan, dtype=np.float32)
    for b in range(K):
        if v_w[b] > 0:
            v_feat[b] = (v_acc[b] / v_w[b]).astype(np.float32)
            vO[b] = 1

    return {
        "bin_edges": edges.astype(np.float64),
        "text_feat": t_feat, "text_P": tP, "text_O": tO,
        "audio_feat": a_feat, "audio_P": aP, "audio_O": aO,
        "vision_feat": v_feat, "vision_P": vP, "vision_O": vO,
    }
