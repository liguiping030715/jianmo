# File inventory

Read-only scan of `workspace/data/raw/`: **246 files**. Full per-file size, SHA-256 for PKL/XLSX, format, media properties, and XLSX fields are in [file_inventory.csv](file_inventory.csv).

Formats: {'': 2, '.mp4': 140, '.xlsx': 2, '.pkl': 102}. Attachment counts: {'other': 2, 'attachment1': 101, 'attachment2': 3, 'attachment3': 60, 'attachment4': 80}.

No original compressed official archive or published reference checksum was present. Archive-level authenticity therefore cannot be independently verified. Per-file PKL/XLSX hashes are recorded for future integrity checks; raw files were not changed.

| Category | Count / fields |
|---|---|
| Attachment 1 | 100 MP4 plus `label-100.xlsx`; 100 rows, columns `video_id, clip_id, text, label, annotation` |
| Attachment 2 | `aligned_50.pkl`, `unaligned_50.pkl`, `label.xlsx`; 4,850 rows and `mode` split column |
| Attachment 3 | 60 PKL: 30 aligned and 30 unaligned, one unlabeled sample per file |
| Attachment 4 | 40 PKL and 40 MP4: 20 paired samples in each version |

PKL fields and counts were inspected directly. See [pkl_schema.md](pkl_schema.md) and [pkl_schema.json](pkl_schema.json).