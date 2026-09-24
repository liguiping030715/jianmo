---
name: problem-analysis
description: Decompose a new mathematical modeling competition problem into deliverables, dependencies, variables, constraints, uncertainty, and a structural task fingerprint before model selection.
---

# Problem Analysis

1. Read the complete statement and every attachment description first.
2. Map every requested action to a deliverable.
3. For each sub-question record:
   - input,
   - requested output,
   - known quantities,
   - unknown/decision variables,
   - constraints,
   - evaluation target,
   - data dependency,
   - dependency on other questions,
   - ambiguity and uncertainty.
4. Separate stated facts from assumptions.
5. Identify hidden coupling between questions.
6. Identify likely intermediate mathematical objects without selecting a detailed algorithm.
7. Create:
   - `workspace/analysis/problem_breakdown.md`
   - `workspace/analysis/variables_and_constraints.md`
   - `workspace/analysis/assumptions.md`
   - `workspace/analysis/task_fingerprint.md`
8. End with a task map, not a final model.

The task fingerprint should describe structure such as temporal/spatial, supervised/unsupervised, physical/data-driven, graph/resource, domain shift, uncertainty, optimization target, and question dependency pattern.
