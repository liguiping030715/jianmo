# Stage 2 readiness

**Stage 2.5 addendum:** Container-level MP4 durations have since been recovered from the supplied files. This does not resolve the frame/feature-to-second map or the critical `P/O` mask blocker; the readiness state below is unchanged. See `workspace/evidence_recovery/timebase_evidence.md` and `cleaning_contract.md`.

**BLOCKED**

The critical valid/padding/missing gate cannot be passed for the full special-test interface. Attachment 3 provides no explicit IDs, valid lengths or observed masks; unaligned Attachment 2 vision lengths contradict nonzero support in hundreds of train/valid cases; terminal zero regions and native visual absence cannot be unambiguously identified as injected gaps. Attachment 3 also omits the precomputed `text` feature used in Attachment 2: aligned special files provide only float32 `text_bert`, whereas unaligned special files provide only `raw_text`. A consistent text input path is technically conceivable, but it has not been verified or chosen in Stage 2. The Q3 media-to-feature timestamp map is also unresolved.

These findings prevent a defensible claim of exact local-gap type/location/duration/ratio or precise source-time evidence under the current interface. A full media decode also found reported frame counts exceeding decodable counts in 122/140 videos; the timebase needs independent verification. Internal train/valid mask-aware pilots may be technically possible, but the reviewer requested the critical gate to block readiness when padding and missingness cannot be reliably separated. No Stage 3 model design or training was started. The next authorized work would be evidence recovery of length/mask generation, special-set text encoding semantics and media timestamps, followed by re-audit; this document does not perform that later stage.

No original compressed archive or published checksum was present, so archive-level checksum verification remains unavailable; individual PKL and XLSX SHA-256 hashes are in the inventory.
