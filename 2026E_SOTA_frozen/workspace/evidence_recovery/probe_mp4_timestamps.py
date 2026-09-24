"""Read-only ISO BMFF timestamp audit for the official MP4 files.

This reads container sample tables, not decoded frame best-effort timestamps.
No raw video is modified and no feature-to-frame mapping is inferred.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path


ROOT = Path("workspace/data/raw")
OUT = Path("workspace/evidence_recovery/media_timestamp_audit.json")


def u32(buf, off):
    return struct.unpack_from(">I", buf, off)[0]


def u64(buf, off):
    return struct.unpack_from(">Q", buf, off)[0]


def i32(buf, off):
    return struct.unpack_from(">i", buf, off)[0]


def i64(buf, off):
    return struct.unpack_from(">q", buf, off)[0]


def boxes(buf, start=0, end=None):
    end = len(buf) if end is None else end
    p = start
    while p + 8 <= end:
        size = u32(buf, p)
        typ = buf[p + 4:p + 8].decode("latin1")
        head = 8
        if size == 1:
            size = u64(buf, p + 8)
            head = 16
        elif size == 0:
            size = end - p
        if size < head or p + size > end:
            raise ValueError(f"invalid box {typ} at {p}, size={size}")
        yield typ, p + head, p + size
        p += size


def child(buf, start, end, name):
    return next(((a, b) for typ, a, b in boxes(buf, start, end) if typ == name), None)


def table_pairs(buf, loc, signed_second=False):
    if loc is None:
        return None
    a, b = loc
    n = u32(buf, a + 4)
    if a + 8 + 8 * n > b:
        raise ValueError("invalid table length")
    return [(u32(buf, a + 8 + 8 * k),
             i32(buf, a + 12 + 8 * k) if signed_second else u32(buf, a + 12 + 8 * k))
            for k in range(n)]


def parse_track(buf, trak, movie_scale):
    ta, tb = trak
    mdia = child(buf, ta, tb, "mdia")
    if mdia is None:
        return None
    ma, mb = mdia
    mdhd = child(buf, ma, mb, "mdhd")
    hdlr = child(buf, ma, mb, "hdlr")
    minf = child(buf, ma, mb, "minf")
    if not (mdhd and hdlr and minf):
        return None
    ha, _ = mdhd
    version = buf[ha]
    timescale = u32(buf, ha + (20 if version else 12))
    duration_ticks = u64(buf, ha + 24) if version else u32(buf, ha + 16)
    kind = buf[hdlr[0] + 8:hdlr[0] + 12].decode("latin1")
    stbl = child(buf, *minf, "stbl")
    if stbl is None:
        return None
    stts = table_pairs(buf, child(buf, *stbl, "stts"))
    ctts_loc = child(buf, *stbl, "ctts")
    ctts = table_pairs(buf, ctts_loc, signed_second=ctts_loc is not None and buf[ctts_loc[0]] == 1)
    stsz = child(buf, *stbl, "stsz")
    if not stts or stsz is None:
        return None
    declared_count = u32(buf, stsz[0] + 8)
    edits = []
    edts = child(buf, ta, tb, "edts")
    if edts:
        elst = child(buf, *edts, "elst")
        if elst:
            ea, eb = elst
            ev = buf[ea]
            n = u32(buf, ea + 4)
            step = 20 if ev else 12
            for k in range(n):
                q = ea + 8 + k * step
                if q + step > eb:
                    raise ValueError("invalid edit table")
                seg = u64(buf, q) if ev else u32(buf, q)
                media = i64(buf, q + 8) if ev else i32(buf, q + 4)
                rate = struct.unpack_from(">hh", buf, q + (16 if ev else 8))
                edits.append({"movie_duration_ticks": seg, "media_start_ticks": media, "rate": rate})
    ticks = []
    dts = 0
    for count, delta in stts:
        for _ in range(count):
            ticks.append(dts)
            dts += delta
    if len(ticks) != declared_count:
        raise ValueError(f"stts count {len(ticks)} != stsz count {declared_count}")
    offsets = []
    if ctts:
        for count, offset in ctts:
            offsets.extend([offset] * count)
        if len(offsets) != len(ticks):
            raise ValueError("ctts count mismatch")
    else:
        offsets = [0] * len(ticks)
    media_pts = [a + b for a, b in zip(ticks, offsets)]
    simple_edit = (not edits or
                   (len(edits) == 1 and edits[0]["media_start_ticks"] >= 0 and edits[0]["rate"] == (1, 0)))
    clip_pts = None
    presented = None
    presented_indices = None
    if simple_edit:
        shift = edits[0]["media_start_ticks"] if edits else 0
        clip_pts = [(p - shift) / timescale for p in media_pts]
        duration_seconds = edits[0]["movie_duration_ticks"] / movie_scale if edits else duration_ticks / timescale
        presented_indices = [i for i, p in enumerate(clip_pts) if 0 <= p < duration_seconds]
        presented = [clip_pts[i] for i in presented_indices]
    return {
        "kind": kind,
        "time_base": f"1/{timescale}",
        "mdhd_duration_seconds": duration_ticks / timescale,
        "sample_count": len(ticks),
        "stts": stts,
        "ctts": ctts,
        "edits": edits,
        "clip_pts_mapping_supported": simple_edit,
        "first_media_pts_ticks": media_pts[0] if media_pts else None,
        "last_media_pts_ticks": media_pts[-1] if media_pts else None,
        "first_clip_pts_seconds": clip_pts[0] if clip_pts else None,
        "last_clip_pts_seconds": clip_pts[-1] if clip_pts else None,
        "edit_presentation_duration_seconds": duration_seconds if simple_edit else None,
        "presented_sample_count": len(presented) if presented is not None else None,
        "first_presented_pts_seconds": min(presented) if presented else None,
        "last_presented_pts_seconds": max(presented) if presented else None,
        "negative_clip_pts_count": sum(p < 0 for p in clip_pts) if clip_pts is not None else None,
        "nonmonotone_decode_order_pts_count": sum(media_pts[k] < media_pts[k-1] for k in range(1, len(media_pts))),
        "media_pts_ticks": media_pts,
        "clip_pts_seconds": clip_pts,
        "presented_sample_indices": presented_indices,
        "presented_pts_seconds": presented,
    }


def parse_file(path):
    buf = path.read_bytes()
    moov = child(buf, 0, len(buf), "moov")
    if moov is None:
        raise ValueError("missing moov")
    mvhd = child(buf, *moov, "mvhd")
    if mvhd is None:
        raise ValueError("missing mvhd")
    a, _ = mvhd
    version = buf[a]
    movie_scale = u32(buf, a + (20 if version else 12))
    movie_duration = u64(buf, a + 24) if version else u32(buf, a + 16)
    tracks = []
    for typ, ta, tb in boxes(buf, *moov):
        if typ == "trak":
            track = parse_track(buf, (ta, tb), movie_scale)
            if track:
                tracks.append(track)
    return {"path": path.as_posix(), "movie_time_base": f"1/{movie_scale}",
            "movie_duration_seconds": movie_duration / movie_scale, "tracks": tracks}


def main():
    paths = sorted(ROOT.rglob("*.mp4"))
    results = []
    errors = []
    for path in paths:
        try:
            results.append(parse_file(path))
        except Exception as exc:
            errors.append({"path": path.as_posix(), "error": repr(exc)})
    OUT.write_text(json.dumps({"method": "ISO BMFF moov/trak/mdhd/stts/ctts/elst/stsz parser",
                               "scope": "container sample timestamps, not decoder best-effort timestamps",
                               "files": results, "errors": errors}, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(json.dumps({"files": len(paths), "parsed": len(results), "errors": errors[:5],
                      "track_kinds": dict(Counter(t["kind"] for r in results for t in r["tracks"])),
                      "unsupported_edits": sum(not t["clip_pts_mapping_supported"] for r in results for t in r["tracks"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
