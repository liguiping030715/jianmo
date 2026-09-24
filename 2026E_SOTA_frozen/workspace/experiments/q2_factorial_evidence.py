"""Inference-only balanced Q2 robustness figures from frozen M0/M1 checkpoints."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
from q2_bridge import NativeMulT
from q2_unaligned_text_adapter import FrozenUnalignedTextAdapter, file_sha256

OUT = Path("workspace/results/q2_evidence")
FIG = Path("workspace/figures")
VALID = Path("workspace/data/processed/cleaning_2026e_v2/unaligned/valid")
CACHE = Path("workspace/experiments/pilot_cache/clean_valid.npy")
RUNS = Path("workspace/experiments/q2_bridge_runs")
FREEZE = Path("workspace/results/q2_final/final_pipeline_freeze.json")
SEEDS = (1729, 2718, 31415)
MODELS = ("M0", "M1")
TYPES = ("text", "audio", "vision", "audio+vision")
POSITIONS = ("early", "middle", "late")
RATIOS = (0.10, 0.20, 0.30)
N = 728


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no records for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preflight():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["selected_seed"] != 2718 or freeze["ensemble"]:
        raise RuntimeError("final Q2 predictor freeze changed")
    hashes = freeze["frozen_hashes_sha256"]
    fixed = {
        "M0_source_q2_bridge_py": Path("q2_bridge.py"),
        "text_adapter_source": Path("q2_unaligned_text_adapter.py"),
        "BERT_weights": Path("workspace/references/bert-base-uncased/pytorch_model.bin"),
        "tokenizer_vocab": Path("workspace/references/bert-base-uncased/vocab.txt"),
        "cleaning_v2_manifest": Path("workspace/data/processed/cleaning_2026e_v2/manifest.json"),
        "unaligned_train_fitted_scaler": Path("workspace/data/processed/cleaning_2026e_v2/unaligned/scalers.json"),
    }
    for name, path in fixed.items():
        if file_sha256(path) != hashes[name]:
            raise RuntimeError(f"frozen source hash mismatch: {name}")
    data = ru.load_split("valid")
    text = np.load(CACHE, mmap_mode="r")
    if len(data["samples"]) != N or text.shape != (N, 50, 768):
        raise RuntimeError("A2 valid count or BERT cache shape mismatch")
    if any(s["row"] != i for i, s in enumerate(data["samples"])):
        raise RuntimeError("A2 valid row order changed")
    token_meta = json.loads((CACHE.parent / "clean_text_cache.json").read_text(encoding="utf-8"))
    import hashlib
    if hashlib.sha256(data["text_bert"].tobytes()).hexdigest() != token_meta["tokens_sha256"]["valid"]:
        raise RuntimeError("A2 valid token triplet source changed")
    loaded = {}
    frozen_metrics = {}
    for model_name in MODELS:
        for seed in SEEDS:
            run = RUNS / f"{model_name}_seed{seed}"
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if (metrics["status"] != "COMPLETE" or
                    file_sha256(run / "checkpoint.pt") != metrics["checkpoint_sha256"]):
                raise RuntimeError(f"frozen checkpoint invalid: {model_name} {seed}")
            model = NativeMulT(model_name == "M1")
            model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
            model.eval()
            loaded[(model_name, seed)] = model
            frozen_metrics[(model_name, seed)] = metrics
    adapter = FrozenUnalignedTextAdapter()
    return data, np.asarray(text), loaded, frozen_metrics, adapter, freeze


def feasible_spans(data):
    """Fixed all-cell common cohort; spans keyed by row/modality/ratio/position."""
    spans = {}
    exclusion = []
    for row in range(N):
        row_spans = {}
        reasons = []
        for modality in ru.MODS:
            eligible = ru.eligible_positions(data, row, modality)
            if len(eligible) < 2:
                reasons.append(f"{modality}:<2 eligible")
                continue
            for ratio in RATIOS:
                width = max(1, int(round(ratio * len(eligible))))
                possible = ru.possible_spans(eligible, width)
                if len(possible) < 3:
                    reasons.append(f"{modality}:{ratio}:<3 feasible starts")
                    continue
                starts = (possible[0], possible[(len(possible)-1)//2], possible[-1])
                if len({x[0] for x in starts}) != 3:
                    reasons.append(f"{modality}:{ratio}:non-distinct anchors")
                    continue
                for position, span in zip(POSITIONS, starts):
                    row_spans[(modality, ratio, position)] = (span, len(eligible))
        if len(row_spans) == len(ru.MODS)*len(RATIOS)*len(POSITIONS) and not reasons:
            spans[row] = row_spans
        else:
            exclusion.append({"row": row, "sample_id": data["samples"][row]["id"],
                              "reasons": "; ".join(reasons)})
    if len(spans) < 650:
        raise RuntimeError("common complete-case cohort unexpectedly small")
    return spans, exclusion


def subset(data, rows):
    out = {"samples": [data["samples"][i] for i in rows]}
    for name in ("text_bert", "text_P", "text_O", "audio", "audio_P", "audio_O",
                 "vision", "vision_P", "vision_O"):
        out[name] = data[name][rows]
    return out


def make_condition(clean, original_rows, spans, missing_type, position, ratio, adapter, clean_text):
    mods = tuple(missing_type.split("+"))
    changed = dict(clean)
    if "text" in mods:
        changed["text_bert"] = clean["text_bert"].copy()
        changed["text_O"] = clean["text_O"].copy()
    for m in ("audio", "vision"):
        if m in mods:
            changed[m] = clean[m].copy()
            changed[f"{m}_O"] = clean[f"{m}_O"].copy()
    for subrow, original_row in enumerate(original_rows):
        for m in mods:
            (start, end), _ = spans[original_row][(m, ratio, position)]
            if m == "text":
                changed["text_bert"][subrow, 0, start:end] = 100
            else:
                changed[m][subrow, start:end] = 0.0
            changed[f"{m}_O"][subrow, start:end] = 0
    states = adapter.encode_triplets(changed["text_bert"]) if "text" in mods else clean_text
    return changed, states


def scores(y_cls, y_reg, logits, intensity):
    if not np.isfinite(logits).all() or not np.isfinite(intensity).all():
        raise RuntimeError("nonfinite factorial prediction")
    pred = logits.argmax(axis=1)
    pearson = (float(np.corrcoef(y_reg, intensity)[0, 1])
               if np.std(y_reg) > 0 and np.std(intensity) > 0 else None)
    return {"accuracy": float(accuracy_score(y_cls, pred)),
            "macro_f1": float(f1_score(y_cls, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_cls, pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "mae": float(mean_absolute_error(y_reg, intensity)), "pearson": pearson}


def existing_three_seed(frozen_metrics):
    per_seed = []
    for model_name in MODELS:
        for seed in SEEDS:
            audit = frozen_metrics[(model_name, seed)]["audit"]
            clean = audit["clean"]
            gaps = [audit["scenarios"][f"S{i}"]["gap_paired"] for i in range(1, 7)]
            per_seed.append({"model": model_name, "seed": seed,
                             "clean_n": clean["n"],
                             "clean_accuracy": clean["accuracy"],
                             "clean_macro_f1": clean["macro_f1"],
                             "clean_weighted_f1": clean["weighted_f1"],
                             "clean_mae": clean["mae"],
                             "clean_pearson": clean["pearson"],
                             "S1_to_S6_mean_macro_f1": float(np.mean([x["macro_f1"] for x in gaps])),
                             "S1_to_S6_mean_mae": float(np.mean([x["mae"] for x in gaps]))})
    aggregate = []
    keys = [k for k in per_seed[0] if k not in ("model", "seed", "clean_n")]
    for model_name in MODELS:
        rows = [x for x in per_seed if x["model"] == model_name]
        record = {"model": model_name, "seeds": "1729,2718,31415", "valid_n": N}
        for key in keys:
            values = [x[key] for x in rows]
            record[f"{key}_mean"] = float(np.mean(values))
            record[f"{key}_sample_sd"] = float(np.std(values, ddof=1))
        aggregate.append(record)
    write_csv(OUT / "m0_m1_three_seed_per_seed.csv", per_seed)
    write_csv(OUT / "m0_m1_three_seed_mean_sd.csv", aggregate)
    return aggregate


def factor_aggregate(cell_rows):
    factors = {"missing_type": list(TYPES), "position": list(POSITIONS),
               "ratio": list(RATIOS)}
    output = []
    for factor, levels in factors.items():
        for level in levels:
            for model_name in MODELS:
                seed_values = []
                for seed in SEEDS:
                    subset_rows = [r for r in cell_rows if r["model"] == model_name and
                                   r["seed"] == seed and r[factor] == level]
                    expected = 9 if factor == "missing_type" else (12 if factor == "position" else 12)
                    if len(subset_rows) != expected:
                        raise RuntimeError("unbalanced factorial marginal")
                    seed_values.append({"delta_f1": float(np.mean([r["delta_macro_f1_clean_minus_gap"] for r in subset_rows])),
                                        "delta_mae": float(np.mean([r["delta_mae_gap_minus_clean"] for r in subset_rows]))})
                output.append({"factor": factor, "level": str(level), "model": model_name,
                               "n_seeds": 3, "n_common_valid_samples": cell_rows[0]["n"],
                               "delta_macro_f1_mean": float(np.mean([x["delta_f1"] for x in seed_values])),
                               "delta_macro_f1_seed_sd": float(np.std([x["delta_f1"] for x in seed_values], ddof=1)),
                               "delta_mae_mean": float(np.mean([x["delta_mae"] for x in seed_values])),
                               "delta_mae_seed_sd": float(np.std([x["delta_mae"] for x in seed_values], ddof=1)),
                               "per_seed_delta_macro_f1": json.dumps([x["delta_f1"] for x in seed_values]),
                               "per_seed_delta_mae": json.dumps([x["delta_mae"] for x in seed_values])})
    write_csv(OUT / "factorial_marginal_mean_sd.csv", output)
    return output


def figures(marginals, cohort_n):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    specs = (("missing_type", list(TYPES), "Synthetic missing modality", "q2_missing_type_performance"),
             ("position", list(POSITIONS), "Native-index interval position", "q2_missing_position_performance"),
             ("ratio", [str(x) for x in RATIOS], "Requested fraction of eligible support", "q2_missing_ratio_duration_performance"))
    files = []
    for factor, levels, xlabel, basename in specs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        x = np.arange(len(levels))
        for model_name, color, offset in (("M0", "#1662a6", -0.12), ("M1", "#ca6c1f", 0.12)):
            rows = {r["level"]: r for r in marginals if r["factor"] == factor and r["model"] == model_name}
            for ax, metric, sd, ylabel in ((axes[0], "delta_macro_f1_mean", "delta_macro_f1_seed_sd", "Macro-F1 degradation (clean − gap)"),
                                           (axes[1], "delta_mae_mean", "delta_mae_seed_sd", "MAE increase (gap − clean)")):
                ax.errorbar(x+offset, [rows[k][metric] for k in levels],
                            yerr=[rows[k][sd] for k in levels], fmt="o-", linewidth=1.8,
                            capsize=3, color=color, label=model_name)
                ax.set_ylabel(ylabel)
                ax.axhline(0, color="#777777", linewidth=.7)
                ax.grid(axis="y", alpha=.25)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{int(float(k)*100)}%" if factor == "ratio" else k for k in levels])
                ax.set_xlabel(xlabel)
        axes[0].legend(frameon=False, loc="best")
        fig.suptitle(f"A2 valid synthetic local gaps (same {cohort_n} samples; mean ± seed SD)")
        for ext in ("png", "pdf"):
            path = FIG / f"{basename}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            files.append({"path": str(path), "sha256": file_sha256(path)})
        plt.close(fig)
    return files


def report(summary, frozen_aggregate, marginals, cell_rows):
    lines = ["# Q2 evidence completion — frozen model robustness", "",
             "**Q2_EVIDENCE_COMPLETE.** The final M0 seed2718 predictor and A3 predictions remain frozen. This is an inference-only A2-valid diagnostic on six frozen M0/M1 checkpoints. No model was trained, selected or changed, and no A2 test / Attachment 3 / Attachment 4 data were accessed.", "",
             f"The balanced type×position×ratio grid has **36 cells × 6 checkpoints**, all on the same **{summary['common_cohort_n']}/{N}** valid samples. {len(summary['excluded_rows'])} rows lack three feasible contiguous native-index starts at every requested ratio; their IDs and reasons are in [factorial_summary.json](factorial_summary.json). The common cohort is a conditional subset; original 728-row validation metrics remain separate.", "",
             "## Paper figures", "",
             "- [Missing modality type → performance](../../figures/q2_missing_type_performance.png)",
             "- [Missing interval position → performance](../../figures/q2_missing_position_performance.png)",
             "- [Missing ratio/native duration → performance](../../figures/q2_missing_ratio_duration_performance.png)", "",
             "Each figure has macro-F1 degradation and MAE increase panels. Positive values mean worse performance. Points are the marginal mean over the other two balanced factors, with sample SD across the three frozen seeds. The ratio denotes fraction of eligible feature positions; neither interval lengths nor positions are physical seconds. M0 and M1 remain separate in every panel.", "",
             "## Frozen M0 versus M1 (full 728-row valid, three seeds)", "",
             "| Model | Clean macro-F1 mean ± SD | Clean MAE mean ± SD | S1–S6 gap macro-F1 mean ± SD | S1–S6 gap MAE mean ± SD |",
             "|---|---:|---:|---:|---:|"]
    for row in frozen_aggregate:
        lines.append(f"| {row['model']} | {row['clean_macro_f1_mean']:.4f} ± {row['clean_macro_f1_sample_sd']:.4f} | {row['clean_mae_mean']:.4f} ± {row['clean_mae_sample_sd']:.4f} | {row['S1_to_S6_mean_macro_f1_mean']:.4f} ± {row['S1_to_S6_mean_macro_f1_sample_sd']:.4f} | {row['S1_to_S6_mean_mae_mean']:.4f} ± {row['S1_to_S6_mean_mae_sample_sd']:.4f} |")
    lines += ["", "The table is recomputed from the six already frozen Bridge metrics files. S1–S6 coupled modality, position and ratio and serve only as model-screening evidence. M1's earlier promotion failure is unchanged; this table does not reselect it.", "",
              "## Interpretation boundary", "",
              "The factorial grid varies one declared design factor while balancing the others on identical samples. It supports conditional model-response comparisons within the tested factor levels; it does not identify real-world causal effects, official Attachment 3 missingness, or an exact seconds-based duration. Text uses `[UNK]` plus frozen BERT; A/V use zeroed feature rows plus synthetic `O=0`. Native zeros and P/O remain intact outside each copied intervention.", "",
              "Machine-readable cells: [factorial_cell_metrics.csv](factorial_cell_metrics.csv); marginals: [factorial_marginal_mean_sd.csv](factorial_marginal_mean_sd.csv); fixed spans: [factorial_mask_manifest.jsonl](factorial_mask_manifest.jsonl); full frozen comparison: [m0_m1_three_seed_mean_sd.csv](m0_m1_three_seed_mean_sd.csv).", ""]
    (OUT / "q2_evidence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if (OUT / "factorial_cell_metrics.csv").exists() or (OUT / "factorial_summary.json").exists():
        raise RuntimeError("Q2 factorial evidence already executed; refusing to overwrite")
    started = time.perf_counter()
    data, text, models, frozen_metrics, adapter, freeze = preflight()
    print("Q2_VALID_FROZEN_PREFLIGHT_PASS", flush=True)
    spans, excluded = feasible_spans(data)
    original_rows = sorted(spans)
    cohort = subset(data, original_rows)
    clean_text = text[original_rows]
    y_cls = np.array([s["classification_label"] for s in cohort["samples"]], dtype=np.int64)
    y_reg = np.array([s["regression_label"] for s in cohort["samples"]], dtype=np.float64)
    if not np.isfinite(y_reg).all() or not np.isin(y_cls, (0, 1, 2)).all():
        raise RuntimeError("invalid valid labels")
    manifest_path = OUT / "factorial_mask_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in original_rows:
            for missing_type in TYPES:
                for position in POSITIONS:
                    for ratio in RATIOS:
                        mods = tuple(missing_type.split("+"))
                        details = {}
                        for m in mods:
                            (lo, hi), eligible_n = spans[row][(m, ratio, position)]
                            details[m] = {"start_inclusive": lo, "end_exclusive": hi,
                                          "native_length": hi-lo, "eligible_count": eligible_n,
                                          "realized_ratio": (hi-lo)/eligible_n,
                                          "relative_native_start": lo/ru.LEN[m],
                                          "relative_native_duration": (hi-lo)/ru.LEN[m]}
                        f.write(json.dumps({"row": row, "sample_id": data["samples"][row]["id"],
                                            "missing_type": missing_type, "position": position,
                                            "requested_ratio": ratio, "spans": details}, ensure_ascii=False)+"\n")
    print(f"COMMON_COHORT={len(original_rows)}; MASK_MANIFEST={len(original_rows)*36}", flush=True)
    baseline = {}
    for key, model in models.items():
        z, y = ru.predict_all(model, cohort, clean_text, None, {}, None)
        baseline[key] = scores(y_cls, y_reg, z, y)
    print("SIX_CLEAN_BASELINES_COMPLETE", flush=True)
    cell_rows = []
    for missing_type in TYPES:
        for position in POSITIONS:
            for ratio in RATIOS:
                changed, states = make_condition(cohort, original_rows, spans,
                                                 missing_type, position, ratio, adapter, clean_text)
                for model_name in MODELS:
                    for seed in SEEDS:
                        key = (model_name, seed)
                        z, y = ru.predict_all(models[key], changed, states, None, {}, None)
                        gap = scores(y_cls, y_reg, z, y)
                        base = baseline[key]
                        cell_rows.append({"model": model_name, "seed": seed,
                                          "missing_type": missing_type, "position": position,
                                          "ratio": ratio, "n": len(original_rows),
                                          **{f"clean_{k}": v for k, v in base.items()},
                                          **{f"gap_{k}": v for k, v in gap.items()},
                                          "delta_macro_f1_clean_minus_gap": base["macro_f1"]-gap["macro_f1"],
                                          "delta_mae_gap_minus_clean": gap["mae"]-base["mae"]})
                print(f"CELL_COMPLETE {missing_type} {position} {ratio:.2f}", flush=True)
    if len(cell_rows) != 36*6 or not all(np.isfinite(r["delta_macro_f1_clean_minus_gap"]) and
                                        np.isfinite(r["delta_mae_gap_minus_clean"]) for r in cell_rows):
        raise RuntimeError("incomplete or nonfinite factorial grid")
    write_csv(OUT / "factorial_cell_metrics.csv", cell_rows)
    marginals = factor_aggregate(cell_rows)
    frozen_aggregate = existing_three_seed(frozen_metrics)
    figures_done = figures(marginals, len(original_rows))
    summary = {"status": "Q2_EVIDENCE_COMPLETE", "scope": "A2 unaligned valid only; frozen-checkpoint inference",
               "common_cohort_n": len(original_rows), "full_valid_n": N,
               "common_cohort_rows": original_rows, "excluded_rows": excluded,
               "factor_levels": {"missing_type": TYPES, "position": POSITIONS, "ratio": RATIOS},
               "cells_per_checkpoint": 36, "checkpoints": [f"{m}_seed{s}" for m in MODELS for s in SEEDS],
               "cell_rows": len(cell_rows), "manifest_rows": len(original_rows)*36,
               "manifest_sha256": file_sha256(manifest_path),
               "cell_metrics_sha256": file_sha256(OUT / "factorial_cell_metrics.csv"),
               "figures": figures_done,
               "frozen_source_hashes": {f"{m}_seed{s}": file_sha256(RUNS / f"{m}_seed{s}" / "checkpoint.pt")
                                        for m in MODELS for s in SEEDS},
               "protocol_sha256": file_sha256(OUT / "factorial_protocol.md"),
               "runtime_seconds": time.perf_counter()-started,
               "no_training_or_model_reselection": True,
               "no_a2_test_a3_a4_access": True}
    (OUT / "factorial_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report(summary, frozen_aggregate, marginals, cell_rows)
    print(json.dumps({"status": summary["status"], "n": len(original_rows),
                      "cell_rows": len(cell_rows), "seconds": summary["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
