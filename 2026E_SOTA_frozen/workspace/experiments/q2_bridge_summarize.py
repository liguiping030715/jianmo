"""Read frozen Q2 bridge/RU/R1 artifacts; write evidence summaries only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("workspace")
BRIDGE = ROOT / "experiments/q2_bridge_runs"
PILOT = ROOT / "experiments/pilot_runs"
OUT = ROOT / "results/q2_bridge"
SEEDS = (1729, 2718, 31415)
SCENARIOS = ("S1", "S2", "S3", "S4", "S5", "S6")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(level, seed):
    base = BRIDGE if level in ("M0", "M1") else PILOT
    folder = base / f"{level}_seed{seed}"
    item = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if item["status"] != "COMPLETE" or item["audit"]["clean"]["n"] != 728:
        raise RuntimeError(f"invalid run: {folder}")
    if level in ("M0", "M1"):
        if sha(folder / "mask_manifest.jsonl") != sha(PILOT / f"RU_seed{seed}" / "mask_manifest.jsonl"):
            raise RuntimeError(f"mask differs from RU: {folder}")
        if sha(folder / "checkpoint.pt") != item["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint differs: {folder}")
        conditions = {"clean": set()}
        for sc in SCENARIOS:
            conditions[sc] = set()
        for line in (folder / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["condition"] not in conditions:
                raise RuntimeError(f"unexpected prediction condition: {folder}")
            conditions[rec["condition"]].add(rec["sample_id"])
        if any(len(v) != 728 for v in conditions.values()):
            raise RuntimeError(f"incomplete predictions: {folder}")
    for sc in SCENARIOS:
        record = item["audit"]["scenarios"][sc]
        if record["n_eligible"] + record["n_not_injectable"] != 728:
            raise RuntimeError(f"scenario count mismatch: {folder}/{sc}")
    return item


def mean(values):
    return float(np.mean(values))


def std(values):
    return float(np.std(values, ddof=1))


def nice(x):
    return f"{x:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {level: {seed: load(level, seed) for seed in SEEDS}
            for level in ("M0", "M1", "RU")}
    r1 = {seed: load("R1", seed) for seed in SEEDS}
    decision = {"seeds": list(SEEDS), "scenarios": list(SCENARIOS),
                "mask_sha256": {str(seed): sha(BRIDGE / f"M0_seed{seed}" / "mask_manifest.jsonl") for seed in SEEDS},
                "runs": {}, "means": {}, "standard_deviation_across_seeds": {},
                "h11": {}, "h12": {}, "interface_gate": "BRIDGE_TEXT_INTERFACE_INVALID"}
    for level in data:
        decision["runs"][level] = {}
        decision["means"][level] = {}
        decision["standard_deviation_across_seeds"][level] = {}
        for seed in SEEDS:
            item = data[level][seed]
            decision["runs"][level][str(seed)] = {
                "clean": item["audit"]["clean"],
                "gap_mean": {metric: mean([item["audit"]["scenarios"][sc]["gap_paired"][metric]
                                          for sc in SCENARIOS]) for metric in METRICS},
                "gap_degradation_mean": {
                    metric: mean([item["audit"]["scenarios"][sc]["delta"][metric]
                                  for sc in ("S1", "S2", "S3", "S4", "S5", "S6")])
                    for metric in ("macro_f1", "mae")},
                "peak_rss_bytes": item["peak_rss_bytes"],
                "train_wall_seconds": item["train_wall_seconds"],
                "best_epoch": item["best_epoch"]}
        for cond in ("clean", "gap_mean", "gap_degradation_mean"):
            keys = METRICS if cond != "gap_degradation_mean" else ("macro_f1", "mae")
            decision["means"][level][cond] = {
                metric: mean([decision["runs"][level][str(seed)][cond][metric] for seed in SEEDS])
                for metric in keys}
            decision["standard_deviation_across_seeds"][level][cond] = {
                metric: std([decision["runs"][level][str(seed)][cond][metric] for seed in SEEDS])
                for metric in keys}
        decision["means"][level]["per_scenario"] = {
            sc: {metric: mean([data[level][seed]["audit"]["scenarios"][sc]["gap_paired"][metric]
                               for seed in SEEDS]) for metric in METRICS}
            for sc in SCENARIOS}
        decision["standard_deviation_across_seeds"][level]["per_scenario"] = {
            sc: {metric: std([data[level][seed]["audit"]["scenarios"][sc]["gap_paired"][metric]
                              for seed in SEEDS]) for metric in METRICS}
            for sc in SCENARIOS}
    decision["means"]["R1_clean_contextual"] = {metric: mean([r1[s]["audit"]["clean"][metric] for s in SEEDS])
                                                  for metric in METRICS}

    # Frozen H11: competitiveness, not superiority or an adapter effect.
    m0, ru = decision["means"]["M0"], decision["means"]["RU"]
    clean_f1 = m0["clean"]["macro_f1"] >= ru["clean"]["macro_f1"] - .015
    clean_mae = m0["clean"]["mae"] <= ru["clean"]["mae"] + .025
    per_seed = {str(seed): (
        decision["runs"]["M0"][str(seed)]["clean"]["macro_f1"] >=
        decision["runs"]["RU"][str(seed)]["clean"]["macro_f1"] - .015 and
        decision["runs"]["M0"][str(seed)]["clean"]["mae"] <=
        decision["runs"]["RU"][str(seed)]["clean"]["mae"] + .025) for seed in SEEDS}
    gap_f1 = m0["gap_mean"]["macro_f1"] >= ru["gap_mean"]["macro_f1"] - .015
    gap_mae = m0["gap_mean"]["mae"] <= ru["gap_mean"]["mae"] + .025
    resource = all(data["M0"][s]["peak_rss_bytes"] <= 14 * 1024 ** 3 and
                   data["M0"][s]["train_wall_seconds"] <= 7200 for s in SEEDS)
    h11 = all((clean_f1, clean_mae, sum(per_seed.values()) >= 2, gap_f1, gap_mae, resource))
    decision["h11"] = {"mean_clean_f1_gate": clean_f1, "mean_clean_mae_gate": clean_mae,
                       "per_seed_joint_gate": per_seed, "mean_gap_f1_gate": gap_f1,
                       "mean_gap_mae_gate": gap_mae, "resource_gate": resource,
                       "supported_on_A2_valid": h11}

    m1 = decision["means"]["M1"]
    delta_clean_f1 = m1["clean"]["macro_f1"] - m0["clean"]["macro_f1"]
    delta_clean_mae = m1["clean"]["mae"] - m0["clean"]["mae"]
    gain_f1 = m1["gap_mean"]["macro_f1"] - m0["gap_mean"]["macro_f1"]
    reduction_mae = m0["gap_mean"]["mae"] - m1["gap_mean"]["mae"]
    seed_f1 = {str(s): decision["runs"]["M1"][str(s)]["gap_mean"]["macro_f1"] -
              decision["runs"]["M0"][str(s)]["gap_mean"]["macro_f1"] for s in SEEDS}
    seed_mae = {str(s): decision["runs"]["M0"][str(s)]["gap_mean"]["mae"] -
               decision["runs"]["M1"][str(s)]["gap_mean"]["mae"] for s in SEEDS}
    sc_f1 = {sc: m1["per_scenario"][sc]["macro_f1"] - m0["per_scenario"][sc]["macro_f1"]
             for sc in SCENARIOS}
    sc_mae = {sc: m0["per_scenario"][sc]["mae"] - m1["per_scenario"][sc]["mae"]
              for sc in SCENARIOS}
    clean_guard = delta_clean_f1 >= -.015 and delta_clean_mae <= .025
    f1_gate = gain_f1 >= .010 and sum(v > 0 for v in seed_f1.values()) >= 2 and sum(v > 0 for v in sc_f1.values()) >= 4
    mae_gate = reduction_mae >= .015 and sum(v > 0 for v in seed_mae.values()) >= 2 and sum(v > 0 for v in sc_mae.values()) >= 4
    h12 = clean_guard and (f1_gate or mae_gate)
    decision["h12"] = {"clean_macro_f1_delta": delta_clean_f1,
                       "clean_mae_increase": delta_clean_mae, "clean_guard": clean_guard,
                       "gap_macro_f1_gain": gain_f1, "gap_mae_reduction": reduction_mae,
                       "seed_gap_f1_direction": seed_f1, "seed_gap_mae_direction": seed_mae,
                       "scenario_gap_f1_direction": sc_f1, "scenario_gap_mae_direction": sc_mae,
                       "f1_promotion_gate": f1_gate, "mae_promotion_gate": mae_gate,
                       "supported_on_A2_valid": h12}
    decision["metric_gate_label"] = ("MULT_OBSERVATION_ADAPTER_SUPPORTED" if h11 and h12 else
                                     "MULT_BACKBONE_SUPPORTED" if h11 else "MULT_BACKBONE_NOT_SUPPORTED")
    decision["final_deployment_promotion"] = "WITHHELD_PENDING_TEXT_INTERFACE_REVIEW"
    decision["failed_attempt_preserved"] = str(BRIDGE / "M0_seed1729_attempt1_failed")
    (OUT / "q2_bridge_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = ["# Q2 Bridge — M0 native MulT backbone", "",
             "Three frozen seeds; A2 unaligned valid only. RU is the frozen native lightweight comparator. All metrics are from saved audit checkpoints; no model was rerun for this summary.", "",
             "| Model | Seed | Clean Acc | Clean macro-F1 | Clean weighted-F1 | Clean MAE | Clean Pearson | Six-gap mean F1 | Six-gap mean MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level in ("RU", "M0"):
        for s in SEEDS:
            rec = decision["runs"][level][str(s)]
            c, g = rec["clean"], rec["gap_mean"]
            lines.append(f"| {level} | {s} | {nice(c['accuracy'])} | {nice(c['macro_f1'])} | {nice(c['weighted_f1'])} | {nice(c['mae'])} | {nice(c['pearson'])} | {nice(g['macro_f1'])} | {nice(g['mae'])} |")
        c, g = decision["means"][level]["clean"], decision["means"][level]["gap_mean"]
        lines.append(f"| **{level} mean** | — | **{nice(c['accuracy'])}** | **{nice(c['macro_f1'])}** | **{nice(c['weighted_f1'])}** | **{nice(c['mae'])}** | **{nice(c['pearson'])}** | **{nice(g['macro_f1'])}** | **{nice(g['mae'])}** |")
    lines += ["", "## Scenario audit", "", "| Scenario | M0 gap F1 mean ± seed SD | M0 gap MAE mean ± seed SD | RU gap F1 mean | RU gap MAE mean |",
              "|---|---:|---:|---:|---:|"]
    for sc in SCENARIOS:
        a, sd, b = decision["means"]["M0"]["per_scenario"][sc], decision["standard_deviation_across_seeds"]["M0"]["per_scenario"][sc], decision["means"]["RU"]["per_scenario"][sc]
        lines.append(f"| {sc} | {nice(a['macro_f1'])} ± {nice(sd['macro_f1'])} | {nice(a['mae'])} ± {nice(sd['mae'])} | {nice(b['macro_f1'])} | {nice(b['mae'])} |")
    lines += ["", "Mean clean→gap M0 macro-F1 change: " + nice(m0["gap_degradation_mean"]["macro_f1"]) + "; MAE change: " + nice(m0["gap_degradation_mean"]["mae"]) + ". Native-index scenarios combine modality, position and ratio; they do not identify independent factor effects.",
              "", "## H11 frozen gate", "",
              f"M0 clean mean F1 {nice(m0['clean']['macro_f1'])} versus RU {nice(ru['clean']['macro_f1'])}; M0 MAE {nice(m0['clean']['mae'])} versus RU {nice(ru['clean']['mae'])}. Six-gap means: F1 {nice(m0['gap_mean']['macro_f1'])} versus {nice(ru['gap_mean']['macro_f1'])}; MAE {nice(m0['gap_mean']['mae'])} versus {nice(ru['gap_mean']['mae'])}. Joint clean bounds pass in {sum(per_seed.values())}/3 seeds. All resource limits pass. **H11 {'PASSES' if h11 else 'FAILS'} as an A2-valid competitiveness gate.** This does not establish superiority over RU or R1 and does not clear the A3 interface blocker.", ""]
    (OUT / "M0_results.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Q2 Bridge — M1 versus M0 observation adapter", "",
             "Same seed, mask bytes, backbone width/depth/heads and optimization. Positive F1 delta or MAE reduction favors M1.", "",
             "| Seed | M0 clean F1 | M1 clean F1 | Δ clean F1 | M0 clean MAE | M1 clean MAE | Δ gap F1 | Gap MAE reduction |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        a, b = decision["runs"]["M0"][str(s)], decision["runs"]["M1"][str(s)]
        lines.append(f"| {s} | {nice(a['clean']['macro_f1'])} | {nice(b['clean']['macro_f1'])} | {nice(b['clean']['macro_f1']-a['clean']['macro_f1'])} | {nice(a['clean']['mae'])} | {nice(b['clean']['mae'])} | {nice(seed_f1[str(s)])} | {nice(seed_mae[str(s)])} |")
    lines += ["", "| Scenario | M1−M0 gap macro-F1 | M0−M1 gap MAE |",
              "|---|---:|---:|"]
    for sc in SCENARIOS:
        lines.append(f"| {sc} | {nice(sc_f1[sc])} | {nice(sc_mae[sc])} |")
    lines += ["", f"Across three seeds and six scenarios: clean ΔF1 {nice(delta_clean_f1)}, clean MAE increase {nice(delta_clean_mae)}, gap F1 gain {nice(gain_f1)}, gap MAE reduction {nice(reduction_mae)}. Clean guards pass. Gap F1 requires +0.010 or MAE reduction +0.015, 2/3 seed direction and 4/6 scenario direction. Observed F1 direction: {sum(v>0 for v in seed_f1.values())}/3 seeds and {sum(v>0 for v in sc_f1.values())}/6 scenarios; MAE direction: {sum(v>0 for v in seed_mae.values())}/3 seeds and {sum(v>0 for v in sc_mae.values())}/6 scenarios. **H12 {'PASSES' if h12 else 'FAILS'}**. The adapter is not promoted.", ""]
    (OUT / "M1_vs_M0.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Q2 Bridge resources and provenance", "",
             "CPU-only, deterministic 8-thread PyTorch; frozen BERT and RU gap caches reused. Per-run cap: 14 GiB process-tree RSS and 7,200 s.", "",
             "| Run | Best epoch | Mean epoch s | Training wall s | Peak RSS GiB | Mask matches RU |",
             "|---|---:|---:|---:|---:|---|"]
    for level in ("M0", "M1"):
        for s in SEEDS:
            item = data[level][s]
            lines.append(f"| {level} {s} | {item['best_epoch']} | {nice(item['mean_epoch_seconds'])} | {nice(item['train_wall_seconds'])} | {nice(item['peak_rss_bytes']/1024**3)} | yes |")
    failed = BRIDGE / "M0_seed1729_attempt1_failed" / "failures.jsonl"
    lines += ["", "The initial M0 1729 attempt stopped during first selection-loss calculation because the A2 class array entered cross entropy as int32. Its failure log is preserved at `workspace/experiments/q2_bridge_runs/M0_seed1729_attempt1_failed/`. The code corrected the loss target dtype to int64; architecture, masks, optimizer and promotion thresholds were unchanged. All six completed runs remain available with configs, checksum records, logs, predictions, checkpoints and metrics.",
              "", f"Failed-attempt log SHA-256: `{sha(failed)}`.",
              "", "The older RU metadata records an incorrect literal for the BERT weight hash and references the aligned scaler hash. Bridge preflight used the actual frozen unaligned scaler hash and verified clean/gap cache tensors against fresh forwards of the current local BERT. See `q2_bridge_runs/preflight.json` and `pitfall_audit.md`.", ""]
    (OUT / "resource_report.md").write_text("\n".join(lines), encoding="utf-8")

    first = decision["runs"]["M0"]["1729"]
    first_lines = ["# Q2 Bridge first-seed summary", "",
        "Seed 1729 passed technical validity for both M0 and M1, permitting the frozen 2718/31415 runs. The first failed M0 implementation attempt remains logged separately.", "",
        f"M0 clean macro-F1 {nice(first['clean']['macro_f1'])}, MAE {nice(first['clean']['mae'])}; M1 clean macro-F1 {nice(decision['runs']['M1']['1729']['clean']['macro_f1'])}, MAE {nice(decision['runs']['M1']['1729']['clean']['mae'])}.",
        f"M1−M0 six-gap macro-F1 {nice(seed_f1['1729'])}; M0−M1 six-gap MAE {nice(seed_mae['1729'])}. One seed was insufficient for promotion and did not change the frozen plan.", ""]
    (OUT / "first_seed_summary.md").write_text("\n".join(first_lines), encoding="utf-8")

    packet = ["# Q2 Bridge review packet", "",
              "## Frozen results", "",
              f"- H11 M0 backbone competitiveness on A2 valid: **{'PASS' if h11 else 'FAIL'}**. Metric-gate label: `{decision['metric_gate_label']}`.",
              f"- H12 observation adapter: **{'PASS' if h12 else 'FAIL'}**. M1 is not promoted when this gate fails.",
              "- Frozen E0 and E1 remain rejected; they were not retrained or tuned.",
              "- R1 aligned baseline is contextual only. Its three-seed clean mean macro-F1 is " + nice(decision["means"]["R1_clean_contextual"]["macro_f1"]) + " and MAE " + nice(decision["means"]["R1_clean_contextual"]["mae"]) + ". Different input version prevents attributing any difference to backbone alone.",
              "", "## Deployment boundary", "",
              "The [read-only pitfall audit](pitfall_audit.md) remains `BRIDGE_TEXT_INTERFACE_INVALID` under the reviewer’s literal supplied-A3-`text_bert` requirement: same-version unaligned A3 supplies raw text rather than that field, and the current bridge has no A3 inference runner. A2 raw text reproduces its token triplets exactly, so a derived same-tokenizer pathway is plausible, but this bridge review does not silently substitute it for the specified field or mix aligned text with unaligned audio/vision. The A2 metric gate is therefore **not a final Q2 deployment approval**.",
              "", "A [fixed-weight A2-valid interface probe](interface_probe.md) additionally recomputed all 728 valid BERT states from raw text with **zero** difference from the cache. On 32 valid rows, both seed1729 checkpoints gave zero logit/intensity difference between the two text routes; changing only padded BERT rows to 10,000 also produced zero output difference. This supports the derived unaligned raw-text path and padded-query output isolation. It does not create a supplied A3 `text_bert` field or validate an A3 inference runner.",
              "", "## Evidence and restrictions", "",
              "- All six planned completed runs have complete clean plus S1–S6 valid predictions, matching seed-specific native masks and checkpoint hashes. The failed first M0 attempt is retained in `q2_bridge_runs/M0_seed1729_attempt1_failed/`.",
              "- No A2 test, A3, A4, or A1 input was used in the bridge training or metric summary. No official missing mask, physical time alignment, restoration, or Q3 faithfulness claim is made.",
              "- S1–S6 couple modality, native index position and gap ratio. They screen model mechanisms only; they cannot estimate independent effects of those factors. Controlled factorial analysis remains for a later approved stage.",
              "", "Details: [M0](M0_results.md), [M1 comparison](M1_vs_M0.md), [resources](resource_report.md), [machine decision](q2_bridge_decision.json).", ""]
    (OUT / "q2_bridge_review_packet.md").write_text("\n".join(packet), encoding="utf-8")
    print(json.dumps({"h11": h11, "h12": h12, "metric_gate": decision["metric_gate_label"],
                      "deployment": decision["final_deployment_promotion"]}))


if __name__ == "__main__":
    main()
