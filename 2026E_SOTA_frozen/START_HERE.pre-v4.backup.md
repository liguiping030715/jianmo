# Start Here

Put the problem statement in `workspace/problem/`.
Put original datasets in `workspace/data/raw/`.

Then open the folder in VS Code + Codex and send:

> Read `AGENTS.md` and relevant skills. Inspect `workspace/problem/` and
> `workspace/data/raw/`. Start only with Stage 1: problem analysis.
> Produce the required analysis files. Do not choose a final model or run
> full experiments yet.

After Stage 1:

> Audit the Stage 1 outputs against the original problem. Then run
> `data-analysis`, propose 2-3 justified candidate models with explicit pilot
> criteria, and stop before full experiments.

After candidates exist:

> Implement the smallest reproducible pilot experiments that can distinguish
> the candidates. Save parameters, metrics and outputs. Then run `reviewer`.

Only after reviewer approval:

> Run full experiments, sensitivity/robustness checks, visualization, and
> draft the paper using only reviewer-approved evidence.
