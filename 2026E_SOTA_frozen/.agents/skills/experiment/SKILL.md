---
name: experiment
description: Implement reproducible pilot and full experiments that test explicit model hypotheses and preserve machine-readable evidence, failures, parameters, and provenance.
---

# Experiment

1. State the hypothesis before running the experiment.
2. Start with the cheapest pilot capable of falsifying a weak model.
3. Compare candidates under the same data split and metrics when possible.
4. Record script/command, inputs, outputs, seed, parameters, environment, runtime, metrics, warnings, and failures.
5. Save machine-readable results in `workspace/results/`.
6. Save experiment records in `workspace/experiments/`.
7. Check feasibility/physical consistency where relevant.
8. Promote to full experiments only after the pilot supports the claimed improvement.
9. Preserve failed runs when they inform model rejection.
