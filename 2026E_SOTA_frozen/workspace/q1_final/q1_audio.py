"""Q1 audio features: openSMILE eGeMAPS low-level descriptors.

One reproducible frame-level acoustic extractor. Every feature frame keeps an
exact clip-relative timestamp. No silent substitution: if openSMILE is
unavailable the pipeline must report blocked rather than swap extractors.
"""
from __future__ import annotations

import numpy as np


def build_smile():
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    return smile, opensmile


def audio_features(smile, pcm16: np.ndarray) -> dict:
    x = pcm16.astype(np.float32) / 32768.0
    df = smile.process_signal(x, 16000).reset_index()
    starts = df["start"].dt.total_seconds().to_numpy()
    ends = df["end"].dt.total_seconds().to_numpy()
    feat = df[smile.feature_names].to_numpy(dtype=np.float32)
    hops = np.diff(starts)
    hop = float(np.median(hops)) if len(hops) else None
    return {
        "frame_start": starts.astype(np.float64),
        "frame_end": ends.astype(np.float64),
        "features": feat,
        "n_frames": int(len(feat)),
        "hop_s": hop,
        "dim": int(feat.shape[1]),
    }


def audio_metadata(opensmile_mod, out: dict) -> dict:
    return {
        "extractor": "openSMILE eGeMAPSv02 LowLevelDescriptors",
        "opensmile_python_version": getattr(opensmile_mod, "__version__", None),
        "sample_rate": 16000,
        "channels": 1,
        "hop_s": out["hop_s"],
        "window_s": 0.02,  # eGeMAPS LLD analysis window (nominal)
        "feature_dimensions": out["dim"],
        "feature_names": list(out["features"].shape and []),
    }
