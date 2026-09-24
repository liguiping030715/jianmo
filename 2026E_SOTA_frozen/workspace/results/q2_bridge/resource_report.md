# Q2 Bridge resources and provenance

CPU-only, deterministic 8-thread PyTorch; frozen BERT and RU gap caches reused. Per-run cap: 14 GiB process-tree RSS and 7,200 s.

| Run | Best epoch | Mean epoch s | Training wall s | Peak RSS GiB | Mask matches RU |
|---|---:|---:|---:|---:|---|
| M0 1729 | 2 | 27.6271 | 145.4108 | 2.8560 | yes |
| M0 2718 | 2 | 25.4961 | 134.2868 | 3.0866 | yes |
| M0 31415 | 2 | 28.2807 | 149.3149 | 3.0907 | yes |
| M1 1729 | 2 | 30.0840 | 157.2693 | 3.0902 | yes |
| M1 2718 | 1 | 28.9674 | 122.9218 | 3.0867 | yes |
| M1 31415 | 2 | 31.7242 | 165.4766 | 3.0888 | yes |

The initial M0 1729 attempt stopped during first selection-loss calculation because the A2 class array entered cross entropy as int32. Its failure log is preserved at `workspace/experiments/q2_bridge_runs/M0_seed1729_attempt1_failed/`. The code corrected the loss target dtype to int64; architecture, masks, optimizer and promotion thresholds were unchanged. All six completed runs remain available with configs, checksum records, logs, predictions, checkpoints and metrics.

Failed-attempt log SHA-256: `0ed11915325d17106bd12fcc0934dae0b66cff5afd02839baa51ad6f887dc367`.

The older RU metadata records an incorrect literal for the BERT weight hash and references the aligned scaler hash. Bridge preflight used the actual frozen unaligned scaler hash and verified clean/gap cache tensors against fresh forwards of the current local BERT. See `q2_bridge_runs/preflight.json` and `pitfall_audit.md`.
