---
name: final-audit
description: Perform the final submission audit across requirements, equations, code, results, figures, references, external tools/models, AI-use disclosure, abstract, and conclusions.
---

# Final Audit

Create `workspace/analysis/final_audit_report.md`.

## Requirement coverage

For each sentence in the problem that requests an action, identify where it is answered and by which evidence/output file.

## Evidence consistency

Check every important number against saved machine-readable results.
Check every figure/table against its generating script and source file.

## Mathematical consistency

Check symbols, dimensions, units, masks, constraints, objectives, boundary conditions, and train/inference interfaces.

## Narrative consistency

Verify that:

- abstract numbers match body results,
- conclusions do not overclaim,
- each question has a direct answer,
- later questions correctly use earlier outputs,
- stated innovations are demonstrated,
- limitations/failure cases are not hidden.

## Reference integrity

Verify that every reference exists and supports the claim attributed to it.
Check that peer-reviewed versions are preferred where available.
Do not allow citation laundering from a secondary summary when a primary source is required.

## Literature-transfer audit

Search for:

- SOTA methods justified only by prestige/newness,
- hidden external-data use,
- unreported setting mismatch,
- current-problem solution material disguised as literature,
- methods added after experiments solely to rationalize results.

## Historical-transfer audit

Search for irrelevant old variable names, old constants, copied wording, unexplained formulas, or methods justified only by precedent.

## Tool / AI / pretrained-model provenance

When required by competition rules, verify the paper/appendix records:

- external tools/models and versions,
- important parameters,
- AI assistance/use,
- human verification,
- official-data compliance.

## Submission status

End with exactly one:

- `SUBMISSION_READY`
- `READY_WITH_MINOR_FIXES`
- `NOT_READY`
