---
name: solution-architecture
description: Design the end-to-end mathematical solution story and intermediate representations connecting sub-questions after current-problem analysis, recent-literature scan, historical retrieval, and negative-transfer audit, before detailed algorithm selection.
---

# Solution Architecture

## Goal

Turn a set of sub-questions into one coherent, evidence-bounded modeling system.

## Required inputs

Read:

- problem breakdown,
- task fingerprint,
- SOTA literature scan/method matrix,
- historical transfer plan,
- negative-transfer audit.

Literature/historical sources may suggest mechanisms. The architecture must still be derived from current-problem needs.

## Workflow

1. For every sub-question define input, output, mathematical object produced, uncertainty, validation criterion, and downstream consumer.
2. Identify which outputs should be reused by later questions.
3. Identify bottleneck representations where one abstraction can simplify multiple questions.
4. Separate domain mechanism from numerical solver/architecture name.
5. For each proposed component cite its evidence source category:
   - current-problem requirement,
   - current-data fact,
   - recent-literature mechanism,
   - historical modeling pattern.
6. Mark unresolved choices that must wait for Stage 2 or a pilot.
7. Draw a dependency map before choosing detailed algorithms.
8. Build a validation blueprint tied to claims, not generic metrics.

## Required outputs

- `workspace/analysis/solution_story.md`
- `workspace/analysis/question_dependency_map.md`
- `workspace/analysis/intermediate_representations.md`
- `workspace/analysis/validation_blueprint.md`

## Reviewer questions

- Are the questions actually connected, or merely described together?
- Does each intermediate object have a clear definition and meaning?
- Is a later question using the strongest justified information from earlier questions?
- Is uncertainty propagated where it matters?
- Could a simpler representation create a cleaner chain?
- Did any SOTA or historical method enter merely because it is fashionable or familiar?
- Are unresolved aligned/unaligned, missingness, calibration, or other interface choices explicitly deferred to evidence rather than guessed?
