"""Read-only Stage 3D decision packet from frozen E1 and comparator records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pilot_stage4 as ps

RESULT = Path("workspace/results/q2_stage3d")
RUNS = Path("workspace/experiments/stage3d_runs")
SCENARIOS = list(ps.SCENARIOS)


def get(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def put(name: str, text: str):
    (RESULT / name).write_text(text, encoding="utf-8")


def fmt(x):
    return "—" if x is None else f"{x:+.4f}"


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    pre = get(RESULT / "frozen_preflight.json")
    cal = get(RESULT / "lambda_calibration.json")
    full = get(RUNS / "E1_full_seed1729/metrics.json")
    local = get(RUNS / "E1_local_only_seed1729/metrics.json")
    consistency = get(RUNS / "E1_consistency_only_seed1729/metrics.json")
    r1 = get(ps.RUN_DIR / "R1_seed1729/metrics.json")
    r0a = get(ps.RUN_DIR / "R0a_seed1729/metrics.json")
    e0 = get(Path("workspace/experiments/stage3c_runs/E0_seed1729/metrics.json"))
    assert all(x["status"] == "COMPLETE" for x in (full, local, consistency, r1, r0a, e0))
    for name, path in (("R1", ps.RUN_DIR / "R1_seed1729/metrics.json"),
                       ("R0a", ps.RUN_DIR / "R0a_seed1729/metrics.json"),
                       ("E0", Path("workspace/experiments/stage3c_runs/E0_seed1729/metrics.json"))):
        assert ps.sha256(path) == pre["comparators"]["1729"][name]["metrics_sha256"]
    assert pre["plan_sha256"] == cal["plan_sha256"]
    required = ("config.json", "mask_manifest.jsonl", "training_log.jsonl",
                "metrics.json", "predictions_valid.jsonl", "failures.jsonl",
                "environment_checksums.json", "checkpoint.pt")
    artifacts = {}
    for mode in ("full", "local_only", "consistency_only"):
        run = RUNS / f"E1_{mode}_seed1729"
        artifacts[mode] = {name: (run / name).exists() for name in required}
        assert all(artifacts[mode].values())
        assert ps.sha256(run / "mask_manifest.jsonl") == pre["mask_sha256"]["1729"]
        assert not (run / "failures.jsonl").read_text(encoding="utf-8").strip()
        assert len((run / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines()) == 728*7
    assert not (RUNS / "E1_full_seed2718").exists()
    assert not (RUNS / "E1_full_seed31415").exists()

    clean = full["audit"]["clean"]
    base = r1["audit"]["clean"]
    d_clean_f1 = clean["macro_f1"] - base["macro_f1"]
    d_clean_mae = clean["mae"] - base["mae"]
    rows = []
    for sc in SCENARIOS:
        a = full["audit"]["scenarios"][sc]
        b = r1["audit"]["scenarios"][sc]
        rows.append({"scenario": sc, "n_eligible": a["n_eligible"],
                     "gap_f1_gain": a["gap_paired"]["macro_f1"] - b["gap_paired"]["macro_f1"],
                     "gap_mae_reduction": b["gap_paired"]["mae"] - a["gap_paired"]["mae"],
                     "f1_degradation_reduction": a["delta"]["macro_f1"] - b["delta"]["macro_f1"],
                     "mae_degradation_reduction": b["delta"]["mae"] - a["delta"]["mae"]})
    mean = {k: float(np.mean([row[k] for row in rows])) for k in
            ("gap_f1_gain", "gap_mae_reduction", "f1_degradation_reduction",
             "mae_degradation_reduction")}
    mech = full["mechanism"]["six_scenario_means"]
    mech_gate = (mech["restoration_gain"] is not None and mech["restoration_gain"] >= .10
                 and mech["representation_gain"] is not None and mech["representation_gain"] >= .10)
    noncat = (d_clean_f1 >= -.05 and d_clean_mae <= .05)
    assert not mech_gate and not noncat
    result = {"status": "Q2_E1_MECHANISM_FAILED", "plan_sha256": pre["plan_sha256"],
              "calibration_sha256": ps.sha256(RESULT / "lambda_calibration.json"),
              "seed1729": {"clean_macro_f1_change_vs_R1": d_clean_f1,
                            "clean_mae_change_vs_R1": d_clean_mae,
                            "scenario_rows": rows, "mean": mean,
                            "restoration_gain": mech["restoration_gain"],
                            "representation_gain": mech["representation_gain"],
                            "mechanism_gate": bool(mech_gate), "noncatastrophic": bool(noncat),
                            "clean_guard_f1": d_clean_f1 >= -.015,
                            "clean_guard_mae": d_clean_mae <= .025},
              "seeds_2718_31415": "NOT_RUN_PER_FROZEN_FIRST_SEED_STOP",
              "artifacts_present": artifacts,
              "frozen_comparator_hashes_unchanged": True,
              "special_test_access": False}
    put("stage3d_decision.json", json.dumps(result, indent=2, ensure_ascii=False))

    table = ["| 情景 | 可注入样本 | gap F1 gain | gap MAE reduction | F1 退化改善 | MAE 退化改善 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        table.append(f"| {r['scenario']} | {r['n_eligible']} | {fmt(r['gap_f1_gain'])} | {fmt(r['gap_mae_reduction'])} | {fmt(r['f1_degradation_reduction'])} | {fmt(r['mae_degradation_reduction'])} |")
    table.append(f"| **六情景均值** | — | **{fmt(mean['gap_f1_gain'])}** | **{fmt(mean['gap_mae_reduction'])}** | **{fmt(mean['f1_degradation_reduction'])}** | **{fmt(mean['mae_degradation_reduction'])}** |")
    comparison = "\n".join([
        "# E1 prediction comparison — Stage 3D", "",
        "All values use A2 aligned_50 valid with frozen seed-1729 audit masks. R1, R0a and E0 were read only; no A2 test/A3/A4 access.", "",
        "| Model | Clean Accuracy | Clean macro-F1 | Clean MAE | Clean Pearson |",
        "|---|---:|---:|---:|---:|",
        *[f"| {name} | {m['audit']['clean']['accuracy']:.4f} | {m['audit']['clean']['macro_f1']:.4f} | {m['audit']['clean']['mae']:.4f} | {m['audit']['clean']['pearson']:.4f} |"
          for name,m in (("R0a (frozen)",r0a),("R1 (frozen)",r1),("E0 (frozen, rejected)",e0),
                         ("E1-full",full),("E1-local-only",local),("E1-consistency-only",consistency))],
        "", "E1-full versus frozen R1: clean ΔF1 " + fmt(d_clean_f1) +
        ", clean ΔMAE " + fmt(d_clean_mae) +
        ". The first-seed non-catastrophic F1 boundary is −0.0500; the final clean guard is −0.0150.",
        "", "## Paired gap comparison: E1-full minus R1", "", *table, "",
        "Positive gap F1 gain and MAE reduction favor E1. Both six-scenario means are negative, so no predictive robustness effect reaches the frozen thresholds (+0.010 F1 or +0.015 MAE).",
        "The 2/3-seed criterion cannot be assessed because the frozen first-seed stop prevents later runs.", "",
        "Run records: `workspace/experiments/stage3d_runs/E1_*_seed1729/`. All three have config, mask copy, epoch log, valid predictions, failure log, environment hashes, checkpoint and metrics.", "",
        "Q2_E1_MECHANISM_FAILED", ""])
    put("E1_comparison.md", comparison)

    ablation = "\n".join([
        "# E1 seed-1729 diagnostic ablations", "",
        "The same frozen training split, mask, optimizer protocol and train-calibrated lambda values were used; disabling a component only sets its weight/path to zero. Ablations cannot promote E1.", "",
        "| Run | Restoration | Local loss | Global consistency | Clean F1 | Clean MAE | Gap F1 gain vs R1 | Gap MAE reduction vs R1 |",
        "|---|---|---|---|---:|---:|---:|---:|",
        *[f"| {label} | {rest} | {loc} | {glob} | {m['audit']['clean']['macro_f1']:.4f} | {m['audit']['clean']['mae']:.4f} | {np.mean([m['audit']['scenarios'][s]['gap_paired']['macro_f1']-r1['audit']['scenarios'][s]['gap_paired']['macro_f1'] for s in SCENARIOS]):+.4f} | {np.mean([r1['audit']['scenarios'][s]['gap_paired']['mae']-m['audit']['scenarios'][s]['gap_paired']['mae'] for s in SCENARIOS]):+.4f} |"
          for label,m,rest,loc,glob in (("E1-full",full,"yes","yes","yes"),
                                        ("E1-local-only",local,"yes","yes","no"),
                                        ("E1-consistency-only",consistency,"no","no","yes"))],
        "", "Local-only has better clean F1 than full (0.5678 versus 0.4908) but still worse gap F1 than R1 by about 0.0606. Consistency-only predicts only 5 neutral cases among 728 clean valid samples and is not a viable rescue. These are diagnostics, not a post-result architecture search.", ""])
    put("E1_ablation_seed1729.md", ablation)

    mech_rows = []
    for sc, v in full["mechanism"]["scenarios"].items():
        local_gain = 1 - v["local_restored_huber"]/v["local_zero_huber"]
        rep_gain = 1 - v["distance_after"]/v["distance_untreated"]
        mech_rows.append(f"| {sc} | {v['n_injected_positions']} | {local_gain:+.3%} | {rep_gain:+.3%} | {v['native_unknown_positions_not_restored']} |")
    mechanism_md = "\n".join([
        "# E1 mechanism diagnostics", "",
        "Gate definitions were frozen in `q2_stage3d_plan.md` before valid predictions. Local gain uses Huber(z_hat,z_full) against Huber(0,z_full); representation gain uses the same E1 weights with versus without injected-position restoration. Both require ≥10% in the equal-scenario mean.", "",
        "| Scenario | Injected latent positions | Local restoration gain | Fused representation gain | Native unknown positions left unrestored |",
        "|---|---:|---:|---:|---:|", *mech_rows, "",
        f"Six-scenario mean latent Huber: restored **{mech['local_restored_huber']:.6f}**, zero reference **{mech['local_zero_huber']:.6f}**; restoration gain **{mech['restoration_gain']:+.2%}** (passes).",
        f"Mean cosine distance: restored **{mech['distance_after']:.6f}**, untreated gap **{mech['distance_untreated']:.6f}**; representation gain **{mech['representation_gain']:+.2%}** (fails).",
        "", "The local target was detached clean latent state at explicit injected positions only. The E1 pilot did not restore native O=2 positions or use zero rows/A3 as target truth.",
        "Train-only calibration: mean prediction/local/global losses " +
        f"{cal['train_loss_means']['prediction']:.6f}/{cal['train_loss_means']['local']:.6f}/{cal['train_loss_means']['global']:.6f}; " +
        f"lambda_local={cal['lambda_local']:.6f}, lambda_global={cal['lambda_global']:.6f}. " +
        "The latter hit the prespecified cap of 10; no valid-driven adjustment was made.", "",
        "Mechanism gate: **FAILED**. A large latent fit against zero did not translate into closer fused clean/gap representations.", ""])
    put("mechanism_diagnostics.md", mechanism_md)

    packet = "\n".join([
        "# Q2 Stage 3D review packet", "",
        "## Frozen boundary", "",
        "E0 remains `E0_REJECTED`. E1 used only frozen A2 aligned_50 train/valid, BERT, scalers, P/O and byte-identical gap manifests. Frozen R0a/R1/E0 metrics and configs matched their preflight SHA-256 values. No A2 test, Attachment 1/3/4, retraining of comparators, or architecture change after seed1729.", "",
        "## Execution", "",
        "The plan was frozen before calibration and valid prediction. Calibration used the first 256 injectable train entries only. E1-full seed1729, E1-local-only seed1729 and E1-consistency-only seed1729 completed; all successes and logs remain in `stage3d_runs/`.", "",
        f"E1-full best epoch {full['best_epoch']}, train wall {full['train_wall_seconds']:.1f}s, peak RAM {full['peak_rss_bytes']/2**30:.2f} GiB; no budget breach.", "",
        "## Frozen gates", "",
        f"- Local restoration gain {mech['restoration_gain']:+.2%} versus required +10%: **pass**.",
        f"- Clean/gap representation-distance gain {mech['representation_gain']:+.2%} versus required +10%: **fail**.",
        f"- First-seed clean macro-F1 Δ versus R1 {d_clean_f1:+.4f} versus non-catastrophic bound −0.0500 and final guard −0.0150: **fail**.",
        f"- Clean MAE Δ {d_clean_mae:+.4f} versus guard +0.0250: **pass**.",
        f"- Six-gap macro-F1 gain {mean['gap_f1_gain']:+.4f}; MAE reduction {mean['gap_mae_reduction']:+.4f}: **neither predictive effect passes**.",
        "", "The mechanism gate and first-seed non-catastrophic rule both fail. Per the preregistered stop, E1-full seeds 2718 and 31415 were **not run**. Three-seed promotion cannot be assessed and E1 cannot be promoted. No parameters, thresholds or masks were changed to rescue the candidate.", "",
        "Details: [comparison](E1_comparison.md), [ablations](E1_ablation_seed1729.md), [mechanism](mechanism_diagnostics.md), [machine decision](stage3d_decision.json).", "",
        "Q2_E1_MECHANISM_FAILED", ""])
    put("q2_stage3d_review_packet.md", packet)
    print(json.dumps({"status": result["status"], "clean_delta_f1": d_clean_f1,
                      "restoration_gain": mech["restoration_gain"],
                      "representation_gain": mech["representation_gain"]}))


if __name__ == "__main__":
    main()
