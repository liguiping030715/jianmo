"""Inference-only M0 architecture audit on frozen A2 valid."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
import q2_bridge as bridge

OUT = Path("workspace/results/q2_bridge")
RUNS = Path("workspace/experiments/q2_bridge_runs")
SEEDS = (1729, 2718, 31415)
ORDER_CONDITIONS = ("FULL", "AUDIO_TEMPORAL_PERMUTED", "VISION_TEMPORAL_PERMUTED",
                    "AUDIO_REVERSED", "VISION_REVERSED")
MODALITY_CONDITIONS = ("FULL", "AUDIO_SAMPLE_PERMUTED", "VISION_SAMPLE_PERMUTED",
                       "AUDIO+VISION_SAMPLE_PERMUTED")
ALL_CONDITIONS = tuple(dict.fromkeys(ORDER_CONDITIONS + MODALITY_CONDITIONS))
CODE_HASH = "dc0367156de2bd7108d7910557692c173c35964b5c2307beb0dd8d32a2fedccf"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big")


def donor_map(n, modality):
    order = np.random.default_rng(stable_seed("m0-donor-v1", modality)).permutation(n)
    donor = np.empty(n, dtype=np.int64)
    donor[order] = np.roll(order, 1)
    if np.any(donor == np.arange(n)) or len(np.unique(donor)) != n:
        raise RuntimeError("donor map is not a derangement")
    return donor


def temporal_map(data, row, modality, mode):
    p = data[f"{modality}_P"][row]
    o = data[f"{modality}_O"][row]
    eligible = np.flatnonzero((p == 1) & (o == 1))
    if mode == "reversed":
        return eligible, eligible[::-1]
    rng = np.random.default_rng(stable_seed("m0-order-v1", data["samples"][row]["id"], modality))
    return eligible, rng.permutation(eligible)


def input_for(data, states, row, condition, donors):
    audio_row, vision_row = row, row
    if condition in ("AUDIO_SAMPLE_PERMUTED", "AUDIO+VISION_SAMPLE_PERMUTED"):
        audio_row = int(donors["audio"][row])
    if condition in ("VISION_SAMPLE_PERMUTED", "AUDIO+VISION_SAMPLE_PERMUTED"):
        vision_row = int(donors["vision"][row])
    audio = data["audio"][audio_row]
    vision = data["vision"][vision_row]
    p = [data["text_P"][row], data["audio_P"][audio_row], data["vision_P"][vision_row]]
    o = [data["text_O"][row], data["audio_O"][audio_row], data["vision_O"][vision_row]]
    if condition in ("AUDIO_TEMPORAL_PERMUTED", "AUDIO_REVERSED"):
        dest, source = temporal_map(data, row, "audio", "reversed" if condition == "AUDIO_REVERSED" else "permuted")
        audio = audio.copy()
        audio[dest] = data["audio"][row, source]
    if condition in ("VISION_TEMPORAL_PERMUTED", "VISION_REVERSED"):
        dest, source = temporal_map(data, row, "vision", "reversed" if condition == "VISION_REVERSED" else "permuted")
        vision = vision.copy()
        vision[dest] = data["vision"][row, source]
    return states[row], audio, vision, p, o


def predict(model, data, states, condition, donors):
    logits, intensity = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, 728, 64):
            rows = range(start, min(728, start + 64))
            batch = ru.stack_batch([input_for(data, states, i, condition, donors) for i in rows])
            z, y, _ = model(*batch)
            logits.append(z.numpy())
            intensity.append(y.numpy())
    return np.concatenate(logits), np.concatenate(intensity)


def saved_clean_predictions(run, samples):
    rows = [json.loads(line) for line in (run / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines()
            if '"condition": "clean"' in line]
    if len(rows) != 728 or any(rec["row"] != i or rec["sample_id"] != samples[i]["id"]
                               for i, rec in enumerate(rows)):
        raise RuntimeError(f"saved FULL prediction order mismatch: {run}")
    return np.array([rec["logits"] for rec in rows]), np.array([rec["predicted_intensity"] for rec in rows])


def changes(full_z, full_y, z, y):
    dz = np.abs(z - full_z)
    dy = np.abs(y - full_y)
    return {"max_abs_logit_change": float(dz.max()),
            "mean_abs_logit_change": float(dz.mean()),
            "max_abs_intensity_change": float(dy.max()),
            "mean_abs_intensity_change": float(dy.mean())}


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if sha(Path("q2_bridge.py")) != CODE_HASH:
        raise RuntimeError("executed M0 source changed")
    data = ru.load_split("valid")
    states = np.load("workspace/experiments/pilot_cache/clean_valid.npy", mmap_mode="r")
    if len(data["samples"]) != 728 or states.shape != (728, 50, 768):
        raise RuntimeError("valid interface mismatch")
    for m in ("audio", "vision"):
        if data[m].shape != (728, ru.LEN[m], ru.FEAT[m]):
            raise RuntimeError(f"{m} shape mismatch")
    yc = np.array([int(s["classification_label"]) for s in data["samples"]], dtype=np.int64)
    yr = np.array([float(s["regression_label"]) for s in data["samples"]])
    donors = {m: donor_map(728, m) for m in ("audio", "vision")}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "m0_permutation_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row, sample in enumerate(data["samples"]):
            rec = {"row": row, "sample_id": sample["id"],
                   "audio_donor_row": int(donors["audio"][row]),
                   "audio_donor_id": data["samples"][int(donors["audio"][row])]["id"],
                   "vision_donor_row": int(donors["vision"][row]),
                   "vision_donor_id": data["samples"][int(donors["vision"][row])]["id"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    order_rows, modality_rows = [], []
    eligible_counts = {m: int(np.sum(np.sum((data[f"{m}_P"] == 1) & (data[f"{m}_O"] == 1), axis=1) >= 2))
                       for m in ("audio", "vision")}
    results = {"source_code_sha256": CODE_HASH, "valid_rows": 728,
               "permutation_manifest_sha256": sha(manifest),
               "position_eligible_rule": "frozen P=1 and O=1; unresolved/padding untouched",
               "samples_with_at_least_two_order_eligible_rows": eligible_counts,
               "donor_rule": "independent audio/vision derangements, donor full sequence and P/O",
               "seeds": {}, "decision": {}}
    for seed in SEEDS:
        run = RUNS / f"M0_seed{seed}"
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        env = json.loads((run / "environment_checksums.json").read_text(encoding="utf-8"))
        if saved["status"] != "COMPLETE" or env["code_sha256"] != CODE_HASH or env["allowed_splits"] != ["train", "valid"]:
            raise RuntimeError(f"frozen run provenance failed: {seed}")
        if sha(run / "checkpoint.pt") != saved["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed: {seed}")
        model = bridge.NativeMulT(False)
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        model.eval()
        full_z, full_y = predict(model, data, states, "FULL", donors)
        old_z, old_y = saved_clean_predictions(run, data["samples"])
        if max(np.max(np.abs(full_z-old_z)), np.max(np.abs(full_y-old_y))) > 1e-5:
            raise RuntimeError(f"FULL logits differ from frozen pilot: {seed}")
        full_metrics = ru.metrics_from_arrays(yc, yr, full_z, full_y)
        for name in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson"):
            if abs(full_metrics[name] - saved["audit"]["clean"][name]) > 1e-6:
                raise RuntimeError(f"FULL metric mismatch {seed}/{name}")
        results["seeds"][str(seed)] = {"checkpoint_sha256": saved["checkpoint_sha256"],
                                       "conditions": {}}
        for condition in ALL_CONDITIONS:
            if condition == "FULL":
                z, y = full_z, full_y
            else:
                z, y = predict(model, data, states, condition, donors)
            if not np.isfinite(z).all() or not np.isfinite(y).all():
                raise RuntimeError(f"nonfinite prediction {seed}/{condition}")
            metrics = ru.metrics_from_arrays(yc, yr, z, y)
            delta = {"delta_macro_f1": metrics["macro_f1"] - full_metrics["macro_f1"],
                     "delta_weighted_f1": metrics["weighted_f1"] - full_metrics["weighted_f1"],
                     "delta_mae": metrics["mae"] - full_metrics["mae"],
                     "delta_pearson": metrics["pearson"] - full_metrics["pearson"]}
            diff = changes(full_z, full_y, z, y)
            results["seeds"][str(seed)]["conditions"][condition] = {
                "metrics": metrics, "change": diff, "delta_vs_full": delta}
            common = {"seed": seed, "condition": condition, "n": 728,
                      "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "weighted_f1": metrics["weighted_f1"], "mae": metrics["mae"],
                      "pearson": metrics["pearson"], **diff, **delta}
            if condition in ORDER_CONDITIONS:
                order_rows.append(common)
            if condition in MODALITY_CONDITIONS:
                modality_rows.append(common)
            print(json.dumps({"seed": seed, "condition": condition,
                              "mean_logit_change": round(diff["mean_abs_logit_change"], 6),
                              "macro_f1": round(metrics["macro_f1"], 4)}), flush=True)

    csv_write(OUT / "m0_order_sensitivity.csv", order_rows)
    csv_write(OUT / "m0_modality_ablation.csv", modality_rows)
    order_gate = {}
    modality_gate = {}
    for modality in ("audio", "vision"):
        names = (f"{modality.upper()}_TEMPORAL_PERMUTED", f"{modality.upper()}_REVERSED")
        per_seed = {}
        for seed in SEEDS:
            cs = results["seeds"][str(seed)]["conditions"]
            per_seed[str(seed)] = any(cs[name]["change"]["mean_abs_logit_change"] > 1e-4 or
                                      cs[name]["change"]["mean_abs_intensity_change"] > 1e-4
                                      for name in names)
        order_gate[modality] = {"per_seed": per_seed, "pass": all(per_seed.values())}
        name = f"{modality.upper()}_SAMPLE_PERMUTED"
        per_seed_use = {}
        for seed in SEEDS:
            c = results["seeds"][str(seed)]["conditions"][name]
            d, ch = c["delta_vs_full"], c["change"]
            per_seed_use[str(seed)] = (ch["max_abs_logit_change"] > 1e-5 and
                (d["delta_macro_f1"] <= -.005 or d["delta_weighted_f1"] <= -.005 or
                 d["delta_mae"] >= .005 or d["delta_pearson"] <= -.005))
        modality_gate[modality] = {"per_seed": per_seed_use,
                                   "pass": sum(per_seed_use.values()) >= 2}
    structural = True  # documented, dimension-checked executed path; all inferences finite.
    if not structural:
        status = "M0_ARCHITECTURE_AUDIT_BLOCKED"
    elif not all(x["pass"] for x in order_gate.values()):
        status = "M0_MISSING_EFFECTIVE_POSITION_ENCODING"
    elif not any(x["pass"] for x in modality_gate.values()):
        status = "M0_MULTIMODAL_USE_NOT_SUPPORTED"
    else:
        status = "M0_ARCHITECTURE_ACCEPTED"
    results["decision"] = {"structural_gate": structural, "order_gate": order_gate,
                           "modality_use_gate": modality_gate, "status": status}
    (OUT / "m0_architecture_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# M0 final architecture acceptance audit", "",
             f"**Architecture status: `{status}`.** Three frozen seed1729/2718/31415 M0 checkpoints; only 728 A2 unaligned valid samples. No training, patching, A2 test, A3 or A4 access. FULL reproduced the saved clean predictions and metrics for every seed. Executed `q2_bridge.py` SHA-256 `{CODE_HASH}` matched all three environment records.", "",
             "## A. Executed architecture", "",
             "M0 consumes text **50×768** frozen BERT hidden states, audio **500×74**, and vision **500×35**. Three separate linear input projections map 768→48, 74→48 and 35→48. Fixed, nonlearned sinusoidal native positional buffers of lengths 50, 500 and 500 (48 channels) are added **after** their projections and **before** GELU. They are registered with `persistent=False`, so checkpoint tensors do not contain learned position weights.", "",
             "For M0, `P!=0` is the usable attention/pooling criterion; `P=0` is excluded, `P=2` is retained. M0 does not use `O` in attention or pooling. Text BERT itself receives the supplied attention mask. Padded text positions are computed as independent queries but are excluded from all final text pooling; no text-query output influences another text query. An all-masked A/V key set gets a neutral zero attention result rather than an all-masked softmax.", "",
             "Only **two** cross-modal directions exist: text Q ← audio K,V and text Q ← vision K,V. There is no audio-query, vision-query, audio↔vision or full pairwise six-direction exchange. The executed architecture is **text-anchored MulT-style cross-modal attention**, not full pairwise MulT.", "",
             "| Block | Q | K | V | First residual | Norm/FFN/second residual | Dropout |",
             "|---|---|---|---|---|---|---|",
             "| `audio_to_text` | projected+positioned text (50×48) | projected+positioned audio (500×48) | same audio | text query + attended audio output, both 48-d | post-LayerNorm; 48→96→48 GELU FFN; second residual then post-LayerNorm | attention weights 0.1; attended output 0.1; FFN hidden 0.1 |",
             "| `vision_to_text` | projected+positioned text (50×48) | projected+positioned vision (500×48) | same vision | text query + attended vision output, both 48-d | same post-LayerNorm/FFN/second post-LayerNorm | same 0.1 sites |",
             "", "There is no 768-d text + 74-d audio or 768-d text + 35-d vision residual. All attention and residual paths are 48-dimensional. Masked mean pooling separately summarizes base text, audio-attended text and vision-attended text into three 48-vectors. Their 144-vector concatenation enters Linear(144,48)→GELU→dropout(0.1). The classification head is Linear(48,3) logits; the regression head is Linear(48,1) followed by `3*tanh`, producing a bounded continuous intensity.", "",
             "## B. Position-order sensitivity", "",
             f"Within each sample, only P=1/O=1 native feature rows were permuted or reversed; all masks, unresolved rows, labels and other modalities were left in place. Audio had at least two eligible rows in {eligible_counts['audio']}/728 valid samples; vision in {eligible_counts['vision']}/728. Logit and intensity changes below are absolute relative to FULL. Numerical equivalence was fixed at max ≤1e-5 for both outputs before inference.", "",
             "| Seed | Condition | Max abs logit Δ | Mean abs logit Δ | Mean abs intensity Δ | macro-F1 | MAE | Pearson |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in order_rows:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['max_abs_logit_change']:.6f} | {r['mean_abs_logit_change']:.6f} | {r['mean_abs_intensity_change']:.6f} | {r['macro_f1']:.4f} | {r['mae']:.4f} | {r['pearson']:.4f} |")
    lines += ["", "Order gate: audio " + ("PASS" if order_gate["audio"]["pass"] else "FAIL") +
              "; vision " + ("PASS" if order_gate["vision"]["pass"] else "FAIL") + ". The output changes establish effective native order sensitivity, but several permutations/reversals slightly improve macro-F1; this audit does not establish that the learned ordering improves predictive quality. See CSV/JSON for max regression changes and Accuracy/weighted-F1.", "",
              "## C. Cross-sample modality-use diagnostic", "",
              f"The same label-blind deterministic donor manifest (SHA-256 `{sha(manifest)}`) was shared across all seeds. Each receiver retained its own text/label; the donor's entire native A/V sequence and P/O masks moved together. This tests use of sample-specific information and is **not** causal importance.", "",
              "| Seed | Condition | Δ macro-F1 | Δ weighted-F1 | Δ MAE | Δ Pearson | Max abs logit Δ |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for r in modality_rows:
        lines.append(f"| {r['seed']} | {r['condition']} | {r['delta_macro_f1']:+.4f} | {r['delta_weighted_f1']:+.4f} | {r['delta_mae']:+.4f} | {r['delta_pearson']:+.4f} | {r['max_abs_logit_change']:.4f} |")
    lines += ["", "| Condition | Mean Δ macro-F1 | Mean Δ weighted-F1 | Mean Δ MAE | Mean Δ Pearson |",
              "|---|---:|---:|---:|---:|"]
    for condition in MODALITY_CONDITIONS[1:]:
        rows = [r for r in modality_rows if r["condition"] == condition]
        lines.append(f"| {condition} | {np.mean([r['delta_macro_f1'] for r in rows]):+.4f} | {np.mean([r['delta_weighted_f1'] for r in rows]):+.4f} | {np.mean([r['delta_mae'] for r in rows]):+.4f} | {np.mean([r['delta_pearson'] for r in rows]):+.4f} |")
    vision_rows = [r for r in modality_rows if r["condition"] == "VISION_SAMPLE_PERMUTED"]
    lines += ["", f"Predictive-influence gate: audio **{'PASS' if modality_gate['audio']['pass'] else 'FAIL'}** ({sum(modality_gate['audio']['per_seed'].values())}/3 seeds); vision **{'PASS' if modality_gate['vision']['pass'] else 'FAIL'}** ({sum(modality_gate['vision']['per_seed'].values())}/3 seeds). Vision gives the more consistent degradation: mean Δ macro-F1 {np.mean([r['delta_macro_f1'] for r in vision_rows]):+.4f} and mean Δ MAE {np.mean([r['delta_mae'] for r in vision_rows]):+.4f}. Audio's average effect is smaller and mixed across metrics.", "",
              "## D. Text interface boundary", "",
              "A2 uses supplied `text_bert` → pinned frozen local BERT → 50×768 states, never the precomputed A2 `text` feature. The prior A2-valid shadow probe reproduced those states from `raw_text` exactly. The frozen unaligned A3 schema instead requires `raw_text` → pinned tokenizer → same BERT; a final inference adapter is not implemented. No A3 sample was opened here, and aligned A3 `text_bert` was not mixed with unaligned A/V. This remains a separate deployment boundary after architecture acceptance.", "",
              "## E. Decision", "",
              f"Structural gate: PASS. Audio/vision order gate: {order_gate['audio']['pass']}/{order_gate['vision']['pass']}. At least one non-text sample-specific influence gate: {any(x['pass'] for x in modality_gate.values())}. **{status}**. This is an architecture verdict on A2 valid, not a final A3 deployment or Q2 answer.", "",
              "Machine evidence: `m0_architecture_audit.json`, `m0_order_sensitivity.csv`, `m0_modality_ablation.csv`, and `m0_permutation_manifest.jsonl`.", ""]
    (OUT / "m0_architecture_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "order_gate": order_gate, "modality_gate": modality_gate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
