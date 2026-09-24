"""Cross-check container edit-list presentation samples with OpenCV FFmpeg decode PTS."""

import json
from pathlib import Path

import cv2


SRC = Path("workspace/evidence_recovery/media_timestamp_audit.json")
OUT = Path("workspace/evidence_recovery/decoded_pts_verification.json")


def main():
    rows = []
    for entry in json.loads(SRC.read_text(encoding="utf-8"))["files"]:
        video = next(t for t in entry["tracks"] if t["kind"] == "vide")
        cap = cv2.VideoCapture(entry["path"], cv2.CAP_FFMPEG)
        if not cap.isOpened():
            rows.append({"path": entry["path"], "error": "decoder failed to open"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        points = []
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            points.append({"pts_frame_units": cap.get(cv2.CAP_PROP_PTS),
                           "pos_msec": cap.get(cv2.CAP_PROP_POS_MSEC)})
        cap.release()
        decoded_seconds = [p["pos_msec"] / 1000 for p in points]
        presented_seconds = sorted(video["presented_pts_seconds"])
        matched_count = len(decoded_seconds) == len(presented_seconds)
        max_origin_relative_error = None
        if matched_count and decoded_seconds:
            max_origin_relative_error = max(abs((d - decoded_seconds[0]) -
                                                (v - presented_seconds[0]))
                                            for d, v in zip(decoded_seconds, presented_seconds))
        rows.append({"path": entry["path"], "backend": "OpenCV CAP_FFMPEG",
                     "fps_reported": fps, "metadata_frame_count": len(video["media_pts_ticks"]),
                     "edit_presented_frame_count": len(presented_seconds),
                     "decoded_frame_count": len(points),
                     "count_matches_edit": matched_count,
                     "max_origin_relative_pts_error_seconds": max_origin_relative_error,
                     "decoded_pts_monotone": all(decoded_seconds[k] >= decoded_seconds[k - 1]
                                                 for k in range(1, len(decoded_seconds))),
                     "first_decoded_pts_seconds": decoded_seconds[0] if decoded_seconds else None,
                     "last_decoded_pts_seconds": decoded_seconds[-1] if decoded_seconds else None,
                     "first_presented_pts_seconds": presented_seconds[0] if presented_seconds else None,
                     "last_presented_pts_seconds": presented_seconds[-1] if presented_seconds else None})
    OUT.write_text(json.dumps({"method": "OpenCV CAP_FFMPEG CAP_PROP_PTS/POS_MSEC cross-check",
                               "files": rows}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"files": len(rows),
                      "failed_open": sum("error" in r for r in rows),
                      "count_match": sum(r.get("count_matches_edit", False) for r in rows),
                      "nonmonotone": sum(not r.get("decoded_pts_monotone", False) for r in rows),
                      "max_relative_error": max((r["max_origin_relative_pts_error_seconds"] or 0)
                                                for r in rows if "error" not in r)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
