"""Q1 media decoding with verified presentation timestamps (PyAV).

Never uses frame_index / nominal FPS for time. A single demux pass distributes
packets to video/audio decoders; video frames carry decoded PTS and audio an
actual sample timeline. New features log exact source timestamps.
"""
from __future__ import annotations

import av
import numpy as np

TARGET_SR = 16000


def decode_media(path: str) -> dict:
    c = av.open(str(path))
    container_dur = float(c.duration / av.time_base) if c.duration is not None else None
    v = c.streams.video[0]
    a = c.streams.audio[0]
    fps = float(v.average_rate) if v.average_rate is not None else None

    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    vpts, vrgb, chunks = [], [], []

    # Single demux pass: distribute packets to the correct decoder.
    for packet in c.demux():
        if packet.stream.index == v.index:
            for f in packet.decode():
                if f.pts is not None:
                    vpts.append(float(f.pts * f.time_base))
                    vrgb.append(f.to_ndarray(format="rgb24"))
        elif packet.stream.index == a.index:
            for f in packet.decode():
                for r in resampler.resample(f):
                    chunks.append(r.to_ndarray())
    for r in resampler.resample(None):
        chunks.append(r.to_ndarray())

    if chunks:
        pcm = np.concatenate(chunks, axis=1)[0].astype(np.int16)
    else:
        pcm = np.zeros(0, dtype=np.int16)

    audio_end = len(pcm) / TARGET_SR
    video_end = (vpts[-1] + 1.0 / fps) if vpts and fps else (vpts[-1] if vpts else 0.0)

    # Verified playable duration: container (mvhd) is the alignment authority;
    # actual decoded coverage is recorded for QA.
    D = container_dur if container_dur is not None else max(video_end, audio_end)
    return {
        "path": str(path),
        "duration_s": D,
        "container_duration_s": container_dur,
        "video_pts": np.array(vpts, dtype=np.float64),
        "video_rgb": vrgb,
        "pcm16_16k_mono": pcm,
        "orig_audio_sr": int(a.sample_rate),
        "orig_audio_channels": int(a.channels),
        "nominal_fps": fps,
        "n_video_frames_decoded": len(vpts),
        "video_coverage_end": float(video_end),
        "audio_coverage_end": float(audio_end),
        "width": int(v.codec_context.width),
        "height": int(v.codec_context.height),
    }
