# 2026E media timebase — Stage 2.5

The 140 official MP4 files were read **without rewriting them**. A reproducible [ISO BMFF sample-table probe](probe_mp4_timestamps.py) extracts `mvhd`, video/audio `mdhd` time bases, `stts` decode times, `ctts` composition offsets, `elst` edits, and sample counts. Its [full machine-readable audit](media_timestamp_audit.json) retains every video-frame and audio-packet media PTS, the applied edit-list presentation times, track time bases and durations. An independent [FFmpeg-backed sequential decoder check](verify_decoded_pts.py) and [results](decoded_pts_verification.json) reads actual frames with OpenCV `CAP_FFMPEG`, `CAP_PROP_PTS`, and `CAP_PROP_POS_MSEC`. This does **not** use `CAP_PROP_FRAME_COUNT/fps` to infer time.

| Verified fact | Attachment 1 | Attachment 4 |
|---|---:|---:|
| Files with both video and audio timestamp tables | 100/100 | 40/40 |
| Video track time base | 83 at 1/15360; 17 at 1/90000 | 36 at 1/15360; 4 at 1/90000 |
| Audio track time base | 100 at 1/44100 | 36 at 1/44100; 2 at 1/48000; 2 at 1/32000 |
| `mvhd` duration range (s) | 2.648–34.567 | 3.700–24.134 |
| Video edit-list presentation duration range (s) | 2.257–29.288 | 3.114–21.348 |
| `mvhd` and video presentation duration differ by >1 ms | 90/100 | 34/40 |
| Raw video sample count differs from edit-presented count | 90/100 | 32/40 |

All 140 MP4s have simple rate-1 edit lists that the probe can map. The first step computes sample `DTS` by cumulative `stts`, then `PTS = DTS + ctts`; track time is `PTS × mdhd_time_base`. The edit list selects presentation samples and maps media time to clip-relative seconds using its `media_start` and `movie_duration` values. B-frame composition order matters: video PTS is nonmonotone in decode/sample order for all 140; presented PTS must be ordered for display. The audio entries are **packet** presentation timestamps with audio track time bases, not decoded waveform sample timestamps. A Q1 waveform extraction must preserve its own audio-sample-to-clip-time mapping.

The decoder opened **140/140** files. Decoded frame count equaled the edit-presented video sample count in **140/140** files, and decoded presentation times were monotone. Once both sequences are referenced to their respective first presented frame, maximum absolute discrepancy between decoded `POS_MSEC` and the sorted edit-presented container PTS was `3.6 × 10^-15` s (floating-point precision). This validates the clip-relative *frame time spacing and ordering* for the available decoder. Example: Attachment 1 `-3g5yACwYnA/13.mp4` has 261 raw video samples but 163 edit-presented and decoded frames; the edited video lasts 5.514 s while `mvhd` says 8.767 s. The old 261-versus-163 finding was a timebase/edit-list issue, not evidence that the video should be deleted.

**Best-effort timestamp terminology.** `CAP_PROP_PTS`/`POS_MSEC` supplies the FFmpeg backend's timestamp for each successfully decoded frame and was checked against container presentation PTS. A standalone `ffprobe` `best_effort_timestamp` field was unavailable in this environment, so that literal field was **not** independently observed. No frame decode errors appeared in this count comparison, but exact behavior of another decoder still needs separate verification. Audio packet PTS are recovered from container tables; decoded audio best-effort/PCM timestamps were not independently checked.

**Source-to-feature boundary.** The MP4 timeline is now defensible for media frames and audio packets. Attachment 2/4 PKLs contain no extractor timestamp table linking their 50/500 feature positions to those frames or packets. Thus a feature interval may be reported by index, and a media frame may be reported by verified clip-relative seconds, but one cannot claim an exact **feature-index → frame/second** mapping from this audit. Do not uniformly distribute 50/500 positions or words over a video duration and call that measured time. The earlier [container-duration note](timebase_evidence.md) is superseded on playable duration by the edit-list evidence here.
