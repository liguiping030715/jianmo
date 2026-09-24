# Question 1 Stage 2 audit scripts

Run from the repository root:

```powershell
python workspace/analysis/scripts/run_stage2_audit.py
```

Optional paths:

```powershell
python workspace/analysis/scripts/run_stage2_audit.py `
  --raw-root workspace/data/raw `
  --output-dir workspace/analysis/audit_artifacts
```

The scripts use only the Python standard library. They:

- read `station_a` and `station_b` without modifying them;
- preserve microwave codes as opaque `q11`–`q14` tokens;
- preserve profiler values as opaque `W1`–`W6` fields;
- preserve every duplicate microwave cycle;
- treat slash-only profiler tokens as structurally missing without imputing them;
- fail on malformed widths, timestamps, numeric tokens, or required markers;
- verify that all raw-file SHA-256 hashes are unchanged during a run;
- write reproducible JSON/CSV evidence to `workspace/analysis/audit_artifacts/`.

`question1_parsers.py` contains the reusable parsers. `run_stage2_audit.py` performs the complete audit and alignment summaries.
