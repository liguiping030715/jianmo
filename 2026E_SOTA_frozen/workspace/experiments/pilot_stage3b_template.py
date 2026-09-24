
"""
Stage 3B pilot runner TEMPLATE.

Important:
- Reuse the existing Stage-4 data loader, BERT cache/intervention code,
  mask manifests, metrics, and logging utilities.
- Do NOT create a second data interpretation.
- This file shows exactly where D0/D1 plug into the proven pipeline.

Suggested integration target:
  workspace/experiments/pilot_stage3b.py

You need to adapt only the function names imported from pilot_stage4.py
to match the repository's existing API.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from stage3b_models import D0TaskSpecificFusion, D1LocalStateTemporalPooling, count_trainable

SEEDS = [1729, 2718, 31415]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def build_model(name: str):
    if name == "D0":
        return D0TaskSpecificFusion(hidden=128, max_len=50)
    if name == "D1":
        return D1LocalStateTemporalPooling(
            hidden=128,
            scorer_hidden=32,
            max_len=50,
            kernel_size=3,
        )
    raise ValueError(name)


def local_weight_diagnostics(output, injected_masks=None):
    """
    Call on D1 batches only.

    Save:
      - entropy of local weights
      - mean weight inside injected interval
      - mean weight just outside gap boundary
      - mean weight on unknown P/O=2
      - position-wise mean weight over dataset
      - seed/scenario variation

    This detects:
      a) gap-boundary response
      b) uniform/static positional collapse
      c) accidental hard exclusion of unknown states

    Never call these weights explanations.
    """
    rows = {}
    eps = 1e-12
    for m in ["text", "audio", "vision"]:
        d = output["diagnostics"][m]
        if "local_weights" not in d:
            continue
        w = d["local_weights"].detach()
        ent = -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(dim=1)
        rows[m] = {
            "entropy_mean": float(ent.mean()),
            "max_weight_mean": float(w.max(dim=1).values.mean()),
            "position_mean": w.mean(dim=0).cpu().tolist(),
        }
    return rows


def main():
    # ------------------------------------------------------------------
    # ADAPT THESE IMPORTS TO THE EXISTING, ALREADY-VALIDATED STAGE-4 API.
    # ------------------------------------------------------------------
    try:
        from pilot_stage4 import (
            load_aligned_split,           # expected: uses frozen v2
            get_or_build_bert_cache,      # keyed by exact token IDs
            build_gap_manifest,           # existing deterministic S1-S6 code
            apply_gap_copy,               # must preserve injected_mask semantics
            train_one_run,
            evaluate_clean_and_scenarios,
        )
    except ImportError as e:
        raise SystemExit(
            "Adapt the six import names at the top of main() to the actual "
            "functions already present in workspace/experiments/pilot_stage4.py. "
            "Do not rewrite the data loader. Original import error: %r" % (e,)
        )

    out_root = Path("workspace/experiments/stage3b_runs")
    out_root.mkdir(parents=True, exist_ok=True)

    # Freeze and save model parameter counts before training.
    counts = {name: count_trainable(build_model(name)) for name in ["D0", "D1"]}
    (out_root / "parameter_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )

    # Exact direct ablations:
    #   frozen R1 -> D0 : only shared fusion -> task-specific fusion
    #   D0 -> D1        : only mean pooling -> local state-aware pooling
    for seed in SEEDS:
        torch.manual_seed(seed)

        train = load_aligned_split("train")
        valid = load_aligned_split("valid")

        # Reuse the already frozen Stage-4 scenario protocol.
        train_manifest = build_gap_manifest(train, split="train", seed=seed)
        valid_manifest = build_gap_manifest(valid, split="valid", seed=seed)

        for model_name in ["D0", "D1"]:
            model = build_model(model_name)

            run_dir = out_root / f"{model_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # train_one_run must reuse the frozen Stage-4:
            # lambda pair, optimizer family, LR search budget, epoch cap,
            # patience, batch size policy, and cache semantics.
            train_one_run(
                model=model,
                train_data=train,
                valid_data=valid,
                train_manifest=train_manifest,
                valid_manifest=valid_manifest,
                seed=seed,
                run_dir=run_dir,
            )

            metrics = evaluate_clean_and_scenarios(
                model=model,
                valid_data=valid,
                valid_manifest=valid_manifest,
                scenarios=SCENARIOS,
                run_dir=run_dir,
            )

            if model_name == "D1":
                # Add diagnostics inside evaluate loop if batch-level outputs are needed.
                # The function above should save local weights per scenario or summary.
                pass

    print("STAGE3B_RUNS_COMPLETE")


if __name__ == "__main__":
    main()
