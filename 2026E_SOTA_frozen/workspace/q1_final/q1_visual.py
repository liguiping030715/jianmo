"""Q1 visual features: MediaPipe face detection + FaceMesh geometry.

Frames come from the verified presentation timeline (decoded PTS). Every
processed frame records decoded index, PTS, detection status, feature vector,
and confidence where available. Frames with no face are kept as an explicit
unavailable observation state, never deleted.
"""
from __future__ import annotations

import numpy as np

N_LANDMARKS = 468


def build_face():
    import mediapipe as mp
    detect = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5)
    return detect, mesh


def visual_features(detect, mesh, rgb_frames, pts) -> dict:
    n = len(rgb_frames)
    feats = np.full((n, N_LANDMARKS * 3), np.nan, dtype=np.float32)
    status = np.zeros(n, dtype=np.int8)       # 0 not_detected, 1 detected
    conf = np.full(n, np.nan, dtype=np.float32)
    for i, rgb in enumerate(rgb_frames):
        d = detect.process(rgb)
        if d.detections:
            status[i] = 1
            conf[i] = float(d.detections[0].score[0])
        m = mesh.process(rgb)
        if m.multi_face_landmarks:
            lm = m.multi_face_landmarks[0].landmark
            feats[i] = np.array(
                [[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)
            status[i] = 1
    return {
        "frame_idx": np.arange(n, dtype=np.int32),
        "pts": np.asarray(pts, dtype=np.float64),
        "status": status,
        "confidence": conf,
        "features": feats,
        "n_frames": int(n),
        "dim": int(N_LANDMARKS * 3),
    }


def visual_metadata(out: dict, width: int, height: int) -> dict:
    detected = int((out["status"] == 1).sum())
    return {
        "extractor": "MediaPipe FaceDetection + FaceMesh(468) static per frame",
        "feature_dimensions": out["dim"],
        "frames_processed": out["n_frames"],
        "frames_face_detected": detected,
        "face_detection_coverage": detected / out["n_frames"] if out["n_frames"] else None,
        "frame_width": width, "frame_height": height,
    }
