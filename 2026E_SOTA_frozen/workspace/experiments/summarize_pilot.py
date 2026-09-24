"""Summarize fixed Stage 4 outputs; no model fitting or special-set access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


RUNS = Path("workspace/experiments/pilot_runs")
RESULT = Path("workspace/results/pilot")
SEEDS = (1729, 2718, 31415)
SCENARIOS = tuple(f"S{i}" for i in range(1, 7))
PREFIXES = ("R0", "R0a", "R1", "R1x", "R2")


def load_run(prefix, seed):
    base = RUNS / f"{prefix}_seed{seed}"
    m = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == "COMPLETE" and "audit" in m
    pred = {}
    for line in (base / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        pred.setdefault(p["condition"], []).append(p)
    assert all(len(pred[c]) == 728 for c in ("clean", *SCENARIOS))
    for c in pred:
        pred[c].sort(key=lambda p: p["row"])
    return m, pred


def load_eligible(seed):
    path = RUNS / f"R1_seed{seed}" / "mask_manifest.jsonl"
    out = {s: np.zeros(728, dtype=bool) for s in SCENARIOS}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["phase"] == "audit":
            out[item["scenario"]][item["row"]] = item["injectable"]
    return out


def source_group(sample_id):
    return sample_id.split("$_$")[0]


def score_rows(rows, idx):
    y = np.array([rows[i]["true_class"] for i in idx])
    p = np.array([rows[i]["predicted_class"] for i in idx])
    yr = np.array([rows[i]["true_intensity"] for i in idx])
    pr = np.array([rows[i]["predicted_intensity"] for i in idx])
    return (float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)),
            float(np.abs(yr - pr).mean()))


def comparison_sample(data, eligible, a, b, drawn_indices):
    # Positive F1 value and positive MAE reduction favor b.
    result = np.zeros(4, dtype=float)
    for seed in SEEDS:
        pred_a, pred_b = data[seed][a][1], data[seed][b][1]
        for scenario in SCENARIOS:
            idx = [i for i in drawn_indices if eligible[seed][scenario][i]]
            if not idx:
                return None
            af, ae = score_rows(pred_a[scenario], idx)
            bf, be = score_rows(pred_b[scenario], idx)
            acf, ace = score_rows(pred_a["clean"], idx)
            bcf, bce = score_rows(pred_b["clean"], idx)
            result += np.array([bf-af, ae-be,
                                (bf-bcf)-(af-acf), (ae-ace)-(be-bce)])
    return result / (len(SEEDS) * len(SCENARIOS))


def bootstrap(data, eligible, a, b, n=400):
    rows = data[SEEDS[0]]["R1"][1]["clean"]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(source_group(row["sample_id"]), []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(20260923)
    results = []
    for _ in range(n):
        drawn_keys = rng.choice(keys, size=len(keys), replace=True)
        indices = [i for key in drawn_keys for i in groups[key]]
        value = comparison_sample(data, eligible, a, b, indices)
        if value is not None:
            results.append(value)
    arr = np.stack(results)
    return {"replicates": len(arr), "lower_90": np.quantile(arr, 0.05, axis=0).tolist(),
            "upper_90": np.quantile(arr, 0.95, axis=0).tolist()}


def compare(data, eligible, a, b):
    seed_rows = []
    for seed in SEEDS:
        ma, mb = data[seed][a][0], data[seed][b][0]
        a_clean, b_clean = ma["audit"]["clean"], mb["audit"]["clean"]
        scenarios = {}
        for scenario in SCENARIOS:
            aa, bb = ma["audit"]["scenarios"][scenario], mb["audit"]["scenarios"][scenario]
            assert aa["n_eligible"] == bb["n_eligible"]
            scenarios[scenario] = {
                "gap_f1_gain": bb["gap_paired"]["macro_f1"] - aa["gap_paired"]["macro_f1"],
                "gap_mae_reduction": aa["gap_paired"]["mae"] - bb["gap_paired"]["mae"],
                "f1_degradation_reduction": bb["delta"]["macro_f1"] - aa["delta"]["macro_f1"],
                "mae_degradation_reduction": aa["delta"]["mae"] - bb["delta"]["mae"],
                "n_eligible": aa["n_eligible"]}
        seed_rows.append({"seed": seed, "clean": {
            "accuracy_change": b_clean["accuracy"] - a_clean["accuracy"],
            "macro_f1_change": b_clean["macro_f1"] - a_clean["macro_f1"],
            "mae_reduction": a_clean["mae"] - b_clean["mae"],
            "pearson_change": (b_clean["pearson"] - a_clean["pearson"])
                              if b_clean["pearson"] is not None and a_clean["pearson"] is not None else None},
                          "scenarios": scenarios,
                          "scenario_mean": {key: float(np.mean([v[key] for v in scenarios.values()]))
                                            for key in ("gap_f1_gain", "gap_mae_reduction",
                                                        "f1_degradation_reduction", "mae_degradation_reduction")}})
    means = {key: float(np.mean([r["scenario_mean"][key] for r in seed_rows]))
             for key in seed_rows[0]["scenario_mean"]}
    return {"comparator": a, "candidate": b, "seed_rows": seed_rows,
            "mean": means, "cluster_bootstrap": bootstrap(data, eligible, a, b)}


def main():
    data = {}
    eligible = {}
    for seed in SEEDS:
        data[seed] = {prefix: load_run(prefix, seed) for prefix in PREFIXES}
        eligible[seed] = load_eligible(seed)
        hashes = {prefix: hashlib.sha256((RUNS / f"{prefix}_seed{seed}" / "mask_manifest.jsonl").read_bytes()).hexdigest()
                  for prefix in PREFIXES}
        assert len(set(hashes.values())) == 1, f"nonidentical masks seed {seed}: {hashes}"
        source_ids = [[r["sample_id"] for r in data[seed][p][1]["clean"]] for p in PREFIXES]
        assert all(x == source_ids[0] for x in source_ids)
    comparisons = {"B1_vs_augmented_B0": compare(data, eligible, "R0a", "R1"),
                   "C1_vs_B1": compare(data, eligible, "R1", "R2"),
                   "B1x_vs_B1": compare(data, eligible, "R1", "R1x")}
    table = {}
    for prefix in PREFIXES:
        table[prefix] = {}
        for seed in SEEDS:
            m = data[seed][prefix][0]["audit"]
            table[prefix][str(seed)] = {
                "clean": m["clean"],
                "mean_gap_macro_f1": float(np.mean([m["scenarios"][s]["gap_paired"]["macro_f1"] for s in SCENARIOS])),
                "mean_gap_mae": float(np.mean([m["scenarios"][s]["gap_paired"]["mae"] for s in SCENARIOS])),
                "mean_delta_f1": float(np.mean([m["scenarios"][s]["delta"]["macro_f1"] for s in SCENARIOS])),
                "mean_delta_mae": float(np.mean([m["scenarios"][s]["delta"]["mae"] for s in SCENARIOS])),
                "peak_rss_bytes": data[seed][prefix][0]["peak_rss_bytes"]}
    result = {"seeds": list(SEEDS), "scenarios": list(SCENARIOS),
              "same_mask_sha_per_seed": {str(seed): hashlib.sha256((RUNS / f"R1_seed{seed}" / "mask_manifest.jsonl").read_bytes()).hexdigest()
                                          for seed in SEEDS},
              "models": table, "comparisons": comparisons,
              "teacher_gate": json.loads((RESULT / "c2_teacher_gate.json").read_text(encoding="utf-8"))}
    (RESULT / "comparison_statistics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, value in comparisons.items():
        print(key, value["mean"], value["cluster_bootstrap"])


if __name__ == "__main__":
    main()
