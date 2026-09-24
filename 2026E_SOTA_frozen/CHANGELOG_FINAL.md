# Final Edition Changelog

Based on `HuaweiCup-Codex-v4-2026-Ready`.

## Added

- Stage 1.1 `Current SOTA Literature Scan`.
- `.agents/skills/sota-literature-scan/SKILL.md`.
- Stage 1.3 `Negative-Transfer & Evidence Audit`.
- `.agents/skills/negative-transfer-audit/SKILL.md`.
- `templates/sota_method_matrix.md`.
- `templates/negative_transfer_audit.md`.
- `templates/ai_usage_log.md`.
- `PROMPT_2026E_START.md`.
- literature/AI provenance and feasibility gates.

## Upgraded

- `AGENTS.md`: current problem -> SOTA -> historical patterns -> transfer audit -> architecture.
- `model-design`: recent literature support + setting mismatch + feasibility + pilot falsification.
- `solution-architecture`: integrates four evidence layers before algorithms.
- `reviewer`: rejects prestige/newness/complexity-driven SOTA transplants.
- `final-audit`: verifies reference integrity, external data/tools/models, and AI-use provenance when required.
- `competition-execution`: time-boxes literature research.
- `sync_jianmo.py`: now also merges the latest derived `workspace/knowledge/` library from the GitHub repository by default.
- workspace reset script: now initializes `evidence/`, `references/`, and `submissions/`.

## Preserved

All original v4 files and historical practice artifacts are preserved. Original `AGENTS.md` and `START_HERE.md` are backed up as `*.pre-final.backup.md`.
