---
name: negative-transfer-audit
description: Audit proposed transfers from recent literature and historical competition cases before solution architecture or model design, rejecting keyword-, prestige-, architecture-, or precedent-driven transplants.
---

# Negative Transfer and Evidence Audit

## Inputs

Read:

- `workspace/analysis/problem_breakdown.md`
- `workspace/analysis/task_fingerprint.md`
- `workspace/references/sota_literature_scan.md`
- `workspace/references/sota_method_matrix.md`
- `workspace/analysis/historical_transfer_plan.md`

## For every candidate transfer

Record:

1. **Structural match** — what current-problem property actually matches?
2. **Current evidence** — what in the official problem/data supports the analogy?
3. **Transferable component** — mechanism/discipline that may transfer.
4. **Non-transferable details** — dataset, constants, architecture, domain assumptions, thresholds, losses, labels, or evaluation setup that must not transfer automatically.
5. **Falsification** — what Stage 2/pilot result would invalidate the transfer?
6. **Feasibility** — data, external-data, compute, dependency, and time requirements.
7. **Integrity/leakage** — any competition-rule or train/validation/test leakage risk.

## Mandatory rejection patterns

Reject a candidate when its main justification is:

- same keyword,
- same broad domain only,
- historical precedent only,
- top-venue prestige,
- newer architecture,
- higher score on another dataset,
- greater complexity,
- a target-specific competition solution.

## Output

Create:

`workspace/analysis/negative_transfer_audit.md`

End each candidate with exactly one status:

- `ALLOW_TO_ARCHITECTURE`
- `ALLOW_COMPONENT_ONLY`
- `HOLD_FOR_DATA_EVIDENCE`
- `REJECT`

A rejected component must not silently reappear in Stage 3.
