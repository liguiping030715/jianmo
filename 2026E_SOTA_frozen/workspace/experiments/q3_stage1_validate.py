"""Frozen Q3 explanation-operator validation on A2 unaligned valid only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
from q2_bridge import NativeMulT
from q2_unaligned_text_adapter import FrozenUnalignedTextAdapter, file_sha256

OUT = Path("workspace/results/q3_stage1")
VALID = Path("workspace/data/processed/cleaning_2026e_v2/unaligned/valid")
CACHE = Path("workspace/experiments/pilot_cache")
RUN = Path("workspace/experiments/q2_bridge_runs/M0_seed2718")
FREEZE = Path("workspace/results/q2_final/final_pipeline_freeze.json")
N = 728
SEED = 1729
SCALES = (0.01, 0.05, 0.10)
TEXT_LENGTHS = (1, 3, 5)
ANCHORS = 12
K = 3
BATCH = 32
CLASS_NAMES = ("Negative", "Neutral", "Positive")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def preflight():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["selected_seed"] != 2718 or freeze["ensemble"]:
        raise RuntimeError("frozen predictor differs")
    hashes = freeze["frozen_hashes_sha256"]
    paths = {
        "checkpoint_M0_seed2718": RUN / "checkpoint.pt",
        "M0_source_q2_bridge_py": Path("q2_bridge.py"),
        "M0_config_seed2718": RUN / "config.json",
        "text_adapter_source": Path("q2_unaligned_text_adapter.py"),
        "tokenizer_vocab": Path("workspace/references/bert-base-uncased/vocab.txt"),
        "BERT_weights": Path("workspace/references/bert-base-uncased/pytorch_model.bin"),
        "cleaning_v2_manifest": Path("workspace/data/processed/cleaning_2026e_v2/manifest.json"),
        "unaligned_train_fitted_scaler": Path("workspace/data/processed/cleaning_2026e_v2/unaligned/scalers.json"),
    }
    for name, path in paths.items():
        if file_sha256(path) != hashes[name]:
            raise RuntimeError(f"frozen checksum mismatch: {name}")
    data = ru.load_split("valid")
    if len(data["samples"]) != N or any(s["row"] != i for i, s in enumerate(data["samples"])):
        raise RuntimeError("valid row order/count mismatch")
    text = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    meta = json.loads((CACHE / "clean_text_cache.json").read_text(encoding="utf-8"))
    if text.shape != (N, 50, 768) or sha_bytes(data["text_bert"].tobytes()) != meta["tokens_sha256"]["valid"]:
        raise RuntimeError("frozen valid BERT cache/source mismatch")
    for m in ru.MODS:
        if not np.isin(data[f"{m}_P"], (0, 1, 2)).all() or not np.isin(data[f"{m}_O"], (0, 1, 2)).all():
            raise RuntimeError(f"invalid {m} state")
    adapter = FrozenUnalignedTextAdapter()
    regenerated = adapter.encode_raw_text([s["raw_text"] for s in data["samples"]])
    if (not np.array_equal(regenerated["text_bert"], data["text_bert"]) or
            not np.array_equal(regenerated["text_hidden"], text) or
            not np.array_equal(regenerated["text_P"], data["text_P"])):
        raise RuntimeError("valid raw_text -> frozen BERT is not cache-equivalent")
    model = NativeMulT(False)
    model.load_state_dict(torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=True))
    model.eval()
    return data, np.asarray(text), adapter, model, freeze


def derangement(n, rng):
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    raise RuntimeError("could not generate no-self donor permutation")


def donors(data, text, model):
    rng = np.random.default_rng(SEED)
    perms = {m: derangement(N, rng) for m in ru.MODS}
    manifest = {"seed": SEED, "rule": "independent no-self permutations; donor full native tensor and P/O",
                "sample_ids_in_recipient_order": [s["id"] for s in data["samples"]],
                "donor_row_by_modality": {m: p.tolist() for m, p in perms.items()}}
    path = OUT / "q3_donor_permutation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    variants = {}
    for label, mods in (("full", ()), ("text", ("text",)), ("audio", ("audio",)),
                        ("vision", ("vision",)), ("audio_vision", ("audio", "vision"))):
        d = dict(data)
        t = text
        if "text" in mods:
            t = text[perms["text"]]
        for m in mods:
            d[f"{m}_P"] = data[f"{m}_P"][perms[m]]
            d[f"{m}_O"] = data[f"{m}_O"][perms[m]]
            if m != "text":
                d[m] = data[m][perms[m]]
        variants[label] = ru.predict_all(model, d, t, None, {}, None)
    return perms, variants, file_sha256(path)


def candidate_grid(data, adapter):
    grids = {m: [[] for _ in range(N)] for m in ru.MODS}
    for row in range(N):
        token_ids = data["text_bert"][row, 0]
        attn = data["text_bert"][row, 1]
        content = np.flatnonzero((attn == 1) & (token_ids != 101) & (token_ids != 102) & (token_ids != 0))
        for width in TEXT_LENGTHS:
            if len(content) < width:
                continue
            for start in np.unique(np.linspace(0, len(content)-width, min(ANCHORS, len(content)-width+1), dtype=int)):
                idx = content[start:start+width]
                toks = adapter.tokenizer.convert_ids_to_tokens(token_ids[idx].tolist())
                grids["text"][row].append({"row": row, "modality": "text", "scale": str(width),
                                            "indices": idx.astype(int).tolist(),
                                            "start": int(idx[0]), "end": int(idx[-1]+1),
                                            "token_text": " ".join(toks),
                                            "phrase_text": adapter.tokenizer.convert_tokens_to_string(toks),
                                            "eligible_count": int(len(content))})
        for m in ("audio", "vision"):
            support = np.flatnonzero((data[f"{m}_P"][row] == 1) & (data[f"{m}_O"][row] == 1))
            if len(support) < 2:
                continue
            for scale in SCALES:
                width = min(len(support)-1, max(1, int(math.ceil(scale * len(support)))))
                for start in np.unique(np.linspace(0, len(support)-width,
                                                   min(ANCHORS, len(support)-width+1), dtype=int)):
                    idx = support[start:start+width]
                    grids[m][row].append({"row": row, "modality": m, "scale": str(scale),
                                          "indices": idx.astype(int).tolist(),
                                          "start": int(idx[0]), "end": int(idx[-1]+1),
                                          "eligible_count": int(len(support))})
    return grids


def forward_tasks(tasks, data, text, adapter, model):
    """Evaluate declared temporary interventions in order; never mutate frozen arrays."""
    if not tasks:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32)
    all_z, all_y = [], []
    for offset in range(0, len(tasks), BATCH):
        chunk = tasks[offset:offset+BATCH]
        triplets = []
        for task in chunk:
            if task["modality"] != "text":
                continue
            row = task["row"]
            token = data["text_bert"][row].copy()
            content = np.flatnonzero((token[1] == 1) & (token[0] != 101) &
                                     (token[0] != 102) & (token[0] != 0))
            chosen = np.array(task["indices"], dtype=np.int64)
            if task.get("mode") == "retain":
                alter = np.setdiff1d(content, chosen, assume_unique=True)
            else:
                alter = np.union1d(chosen, np.array(task.get("base_indices", []), dtype=np.int64))
            token[0, alter] = 100
            triplets.append(token)
        encoded = adapter.encode_triplets(np.stack(triplets)) if triplets else None
        t_cursor = 0
        parts = []
        for task in chunk:
            row, m = task["row"], task["modality"]
            t = encoded[t_cursor] if m == "text" else text[row]
            if m == "text":
                t_cursor += 1
            p = [data[f"{name}_P"][row].copy() for name in ru.MODS]
            o = [data[f"{name}_O"][row] for name in ru.MODS]
            if m != "text":
                chosen = np.array(task["indices"], dtype=np.int64)
                if task.get("mode") == "retain":
                    known = np.flatnonzero((p[ru.MODS.index(m)] == 1) & (o[ru.MODS.index(m)] == 1))
                    alter = np.setdiff1d(known, chosen, assume_unique=True)
                else:
                    alter = np.union1d(chosen, np.array(task.get("base_indices", []), dtype=np.int64))
                p[ru.MODS.index(m)][alter] = 0
                if not np.any(p[ru.MODS.index(m)] != 0):
                    raise RuntimeError("local intervention would remove every cross-attention key")
            parts.append((t, data["audio"][row], data["vision"][row], p, o))
        with torch.inference_mode():
            z, y, _ = model(*ru.stack_batch(parts))
        all_z.append(z.numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_z), np.concatenate(all_y)


def responses(tasks, z, y, full_z, full_y):
    rows = np.fromiter((task["row"] for task in tasks), dtype=np.int64)
    cls = full_z.argmax(axis=1)[rows]
    dcls = full_z[rows, cls] - z[np.arange(len(tasks)), cls]
    dreg = np.abs(full_y[rows] - y)
    return dcls, dreg


def local_evidence(grids, data, text, adapter, model, full_z, full_y):
    results = {}
    for m in ru.MODS:
        tasks = [candidate for row in grids[m] for candidate in row]
        print(f"LOCAL_{m.upper()}_CANDIDATES={len(tasks)}", flush=True)
        z, y = forward_tasks(tasks, data, text, adapter, model)
        dc, dr = responses(tasks, z, y, full_z, full_y)
        table = []
        for j, candidate in enumerate(tasks):
            row = candidate["row"]
            record = {"row": row, "sample_id": data["samples"][row]["id"],
                      "predicted_class_index": int(full_z[row].argmax()),
                      "modality": m, "scale": candidate["scale"],
                      "start_index_inclusive": candidate["start"],
                      "end_index_exclusive": candidate["end"],
                      "selected_eligible_positions": len(candidate["indices"]),
                      "eligible_positions_total": candidate["eligible_count"],
                      "relative_start_native": candidate["start"] / (50 if m == "text" else 500),
                      "relative_duration_native": (candidate["end"]-candidate["start"]) / (50 if m == "text" else 500),
                      "relative_support_fraction": len(candidate["indices"])/candidate["eligible_count"],
                      "token_text": candidate.get("token_text", ""),
                      "phrase_text": candidate.get("phrase_text", ""),
                      "Delta_cls": float(dc[j]), "Delta_reg": float(dr[j]),
                      "intervened_predicted_class_logit": float(z[j, full_z[row].argmax()]),
                      "intervened_intensity": float(y[j]),
                      "time_seconds": "", "certified_missingness": False}
            table.append(record)
            candidate["Delta_cls"] = float(dc[j])
            candidate["Delta_reg"] = float(dr[j])
        path = OUT / f"q3_local_{m}_evidence.csv"
        write_csv(path, table)
        results[m] = {"candidate_count": len(tasks), "sha256": file_sha256(path),
                      "Delta_cls": describe(dc), "Delta_reg": describe(dr)}
    return results


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty CSV requested: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def choose_disjoint(candidates, order, k):
    chosen = []
    occupied = set()
    for j in order:
        cand = candidates[int(j)]
        position_set = set(cand["indices"])
        if not occupied.intersection(position_set):
            chosen.append(cand)
            occupied.update(position_set)
            if len(chosen) == k:
                break
    return chosen if len(chosen) == k else []


def group_task(row, modality, selected, mode):
    indices = sorted(set().union(*(set(c["indices"]) for c in selected)))
    return {"row": row, "modality": modality, "indices": indices, "mode": mode}


def bootstrap_pair(difference, seed):
    d = np.asarray(difference, dtype=np.float64)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(1000, dtype=np.float64)
    for b in range(1000):
        means[b] = d[rng.integers(0, len(d), len(d))].mean()
    return {"mean": float(d.mean()), "ci95": [float(x) for x in np.quantile(means, [.025, .975])]}


def faithfulness(grids, data, text, adapter, model, full_z, full_y):
    result = {}
    for m in ru.MODS:
        result[m] = {}
        family = "3" if m == "text" else "0.05"
        for target in ("cls", "reg"):
            tasks = []
            keys = []
            skips = Counter()
            for row in range(N):
                cands = [c for c in grids[m][row] if c["scale"] == family]
                if len(cands) < K:
                    skips["fewer_than_3_candidates"] += 1
                    continue
                vals = np.array([c[f"Delta_{target}"] for c in cands])
                top = choose_disjoint(cands, np.argsort(-vals, kind="stable"), K)
                bottom = choose_disjoint(cands, np.argsort(vals, kind="stable"), K)
                rng = np.random.default_rng(SEED + row*1009 + ru.MODS.index(m)*100003 + (0 if target == "cls" else 1))
                random = choose_disjoint(cands, rng.permutation(len(cands)), K)
                if not top or not bottom or not random:
                    skips["no_three_disjoint_equal_width_groups"] += 1
                    continue
                if any(len(set().union(*(set(c["indices"]) for c in selected))) >= cands[0]["eligible_count"]
                       for selected in (top, random, bottom)):
                    skips["joint_deletion_would_remove_all_known_support"] += 1
                    continue
                for label, selected in (("top", top), ("random", random), ("bottom", bottom)):
                    for mode in ("delete", "retain"):
                        keys.append((row, label, mode))
                        tasks.append(group_task(row, m, selected, mode))
            print(f"FAITHFULNESS_{m.upper()}_{target.upper()}_TASKS={len(tasks)}", flush=True)
            if not tasks:
                result[m][target] = {"eligible_samples": 0, "skips": dict(skips), "passes": False}
                continue
            z, y = forward_tasks(tasks, data, text, adapter, model)
            dc, dr = responses(tasks, z, y, full_z, full_y)
            values = defaultdict(dict)
            for j, (row, label, mode) in enumerate(keys):
                values[row][f"{label}_{mode}"] = float(dc[j] if target == "cls" else dr[j])
            order = ("top", "random", "bottom")
            deletion = {label: np.array([values[row][f"{label}_delete"] for row in sorted(values)]) for label in order}
            retention = {label: np.array([values[row][f"{label}_retain"] for row in sorted(values)]) for label in order}
            tr = bootstrap_pair(deletion["top"]-deletion["random"], SEED+11+ru.MODS.index(m)*2+(target=="reg"))
            rb = bootstrap_pair(deletion["random"]-deletion["bottom"], SEED+31+ru.MODS.index(m)*2+(target=="reg"))
            passed = bool(tr["ci95"][0] > 0 and rb["ci95"][0] > 0)
            result[m][target] = {"eligible_samples": len(values), "skips": dict(skips),
                                 "joint_deletion": {label: describe(deletion[label]) for label in order},
                                 "joint_retention_response": {label: describe(retention[label]) for label in order},
                                 "paired_top_minus_random": tr,
                                 "paired_random_minus_bottom": rb,
                                 "passes_operator_consistency": passed,
                                 "per_sample_deletion": {str(row): {label: values[row][f"{label}_delete"] for label in order} for row in sorted(values)},
                                 "per_sample_retention": {str(row): {label: values[row][f"{label}_retain"] for label in order} for row in sorted(values)}}
    return result


def stability(grids, data, text, adapter, model, full_z, full_y):
    selected_rows = np.unique(np.linspace(0, N-1, 128, dtype=int))
    result = {}
    for m in ru.MODS:
        result[m] = {}
        family = "3" if m == "text" else "0.05"
        for target in ("cls", "reg"):
            candidates = {}
            base_tasks = []
            reasons = Counter()
            for row in selected_rows:
                row = int(row)
                cands = [c for c in grids[m][row] if c["scale"] == family]
                if len(cands) < 2:
                    reasons["fewer_than_2_candidates"] += 1
                    continue
                bottom = min(cands, key=lambda c: c[f"Delta_{target}"])
                candidates[row] = (cands, bottom)
                base_tasks.append({"row": row, "modality": m, "indices": bottom["indices"], "mode": "delete"})
            bz, by = forward_tasks(base_tasks, data, text, adapter, model)
            preserved = []
            for j, task in enumerate(base_tasks):
                row = task["row"]
                if (int(bz[j].argmax()) == int(full_z[row].argmax()) and
                        abs(float(by[j]-full_y[row])) <= 0.1):
                    preserved.append((row, bz[j], by[j]))
                else:
                    reasons["prediction_not_preserved"] += 1
            test_tasks = []
            entries = []
            for row, base_z, base_y in preserved:
                cands, bottom = candidates[row]
                for j, cand in enumerate(cands):
                    if cand is bottom:
                        continue
                    task = {"row": row, "modality": m, "indices": cand["indices"],
                            "base_indices": bottom["indices"], "mode": "delete"}
                    test_tasks.append(task)
                    entries.append((row, j, base_z, base_y))
            if not test_tasks:
                result[m][target] = {"selected_rows": len(selected_rows), "eligible_rows": 0,
                                     "excluded": dict(reasons), "top1_agreement": None, "mean_top3_jaccard": None}
                continue
            tz, ty = forward_tasks(test_tasks, data, text, adapter, model)
            by_row = defaultdict(dict)
            for q, (row, j, base_z, base_y) in enumerate(entries):
                cls = int(full_z[row].argmax())
                score = float(base_z[cls]-tz[q, cls]) if target == "cls" else abs(float(base_y-ty[q]))
                by_row[row][j] = score
            top1 = []
            jac = []
            for row, scores in by_row.items():
                cands, bottom = candidates[row]
                pairs = [(j, c) for j, c in enumerate(cands) if c is not bottom]
                original = sorted(pairs, key=lambda v: (-v[1][f"Delta_{target}"], v[0]))
                perturbed = sorted(pairs, key=lambda v: (-scores[v[0]], v[0]))
                if not original or not perturbed:
                    continue
                top1.append(original[0][0] == perturbed[0][0])
                a = {x[0] for x in original[:3]}
                b = {x[0] for x in perturbed[:3]}
                jac.append(len(a & b) / len(a | b))
            result[m][target] = {"selected_rows": len(selected_rows), "eligible_rows": len(top1),
                                 "excluded": dict(reasons), "top1_agreement": float(np.mean(top1)) if top1 else None,
                                 "mean_top3_jaccard": float(np.mean(jac)) if jac else None,
                                 "perturbation": "lowest-response same-scale local interval; predicted class unchanged and |delta intensity| <= 0.1"}
            print(f"STABILITY_{m.upper()}_{target.upper()}={len(top1)}", flush=True)
    return result


def report(payload):
    status = payload["status"]
    faith = payload["faithfulness"]
    lines = ["# Q3 Stage 1 explanation-method review", "",
             f"**Decision: {status}.** Frozen M0 seed2718, A2 valid 728 only. No model update or special/test access. The operator and acceptance rule were written in [q3_explanation_method.md](q3_explanation_method.md) before scoring.", "",
             "## Modality donor responses", "",
             "The [per-sample table](q3_modality_contribution.csv) contains the original predicted-class logit response and absolute regression response to deterministic cross-sample donor replacement. These are model intervention responses, not causal effects. Classification and regression scores and dominant modalities are separate.", "",
             "## Local evidence and faithfulness", "",
             "Local [text](q3_local_text_evidence.csv), [audio](q3_local_audio_evidence.csv), and [vision](q3_local_vision_evidence.csv) tables give every evaluated candidate interval. Audio/vision use native indices only; seconds are unavailable. Text phrases are tokenizer reconstructions, not verified spoken timestamps.", "",
             "| Modality | Target | Eligible | Top mean | Random mean | Bottom mean | Top–random 95% CI | Random–bottom 95% CI | Pass |",
             "|---|---|---:|---:|---:|---:|---|---|---|"]
    for m in ru.MODS:
        for target in ("cls", "reg"):
            d = faith[m][target]
            if d["eligible_samples"]:
                means = [d["joint_deletion"][x]["mean"] for x in ("top", "random", "bottom")]
                ci1 = d["paired_top_minus_random"]["ci95"]
                ci2 = d["paired_random_minus_bottom"]["ci95"]
                lines.append(f"| {m} | {target} | {d['eligible_samples']} | {means[0]:.6f} | {means[1]:.6f} | {means[2]:.6f} | [{ci1[0]:.6f}, {ci1[1]:.6f}] | [{ci2[0]:.6f}, {ci2[1]:.6f}] | {d['passes_operator_consistency']} |")
            else:
                lines.append(f"| {m} | {target} | 0 | — | — | — | — | — | False |")
    lines += ["", "The top/random/bottom test uses joint equal-count deletions after ranking individual intervals on the same valid rows. It checks internal consistency of the declared operator; in-sample ranking and reference choice limit external interpretation. Retention responses and skip counts are in [q3_faithfulness.json](q3_faithfulness.json).", "",
              "## Stability", "",
              "| Modality | Target | Prediction-preserving rows | Top-1 agreement | Mean top-3 Jaccard |",
              "|---|---|---:|---:|---:|"]
    for m in ru.MODS:
        for target in ("cls", "reg"):
            s = payload["stability"][m][target]
            lines.append(f"| {m} | {target} | {s['eligible_rows']} | {s['top1_agreement'] if s['top1_agreement'] is not None else '—'} | {s['mean_top3_jaccard'] if s['mean_top3_jaccard'] is not None else '—'} |")
    lines += ["", "The stability perturbation is a small declared masking or `[UNK]` intervention. Only unchanged-class, ≤0.1-intensity-change cases enter its overlap estimate; excluded counts remain in JSON.", "",
              "## Provenance and limits", "",
              f"Checkpoint SHA-256 `{payload['checkpoint_sha256']}`; donor manifest SHA-256 `{payload['donor_manifest_sha256']}`. Runtime {payload['runtime_seconds']:.1f} s. No attention-weight attribution, certified hidden missing mask, physical-second localization, or causal claim is made.", ""]
    (OUT / "q3_stage1_review.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    required = [OUT / name for name in ("q3_modality_contribution.csv", "q3_local_text_evidence.csv",
                                      "q3_local_audio_evidence.csv", "q3_local_vision_evidence.csv",
                                      "q3_faithfulness.json", "q3_stage1_review.md")]
    if any(p.exists() for p in required):
        raise RuntimeError("Q3 Stage 1 outputs already exist; refusing to overwrite")
    start = time.perf_counter()
    data, text, adapter, model, freeze = preflight()
    print("PREFLIGHT_VALID_728_PASS", flush=True)
    perms, predictions, manifest_hash = donors(data, text, model)
    full_z, full_y = predictions["full"]
    saved = [json.loads(line) for line in (RUN / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines()]
    saved_clean = [r for r in saved if r["condition"] == "clean"]
    if (len(saved_clean) != N or any(r["row"] != i for i, r in enumerate(saved_clean)) or
            np.max(np.abs(full_z-np.array([r["logits"] for r in saved_clean]))) > 1e-5 or
            np.max(np.abs(full_y-np.array([r["predicted_intensity"] for r in saved_clean]))) > 1e-5):
        raise RuntimeError("full valid prediction differs from frozen Bridge evidence")
    cls = full_z.argmax(axis=1)
    rows = []
    donor_summary = {}
    for m in ("text", "audio", "vision", "audio_vision"):
        z, y = predictions[m]
        donor_summary[m] = {"Delta_cls": describe(full_z[np.arange(N), cls]-z[np.arange(N), cls]),
                            "Delta_reg": describe(np.abs(full_y-y))}
    for i, sample in enumerate(data["samples"]):
        record = {"row": i, "sample_id": sample["id"], "predicted_class_index": int(cls[i]),
                  "predicted_polarity": CLASS_NAMES[int(cls[i])], "full_predicted_class_logit": float(full_z[i, cls[i]]),
                  "full_intensity": float(full_y[i])}
        for m in ("text", "audio", "vision"):
            z, y = predictions[m]
            record[f"donor_{m}_row"] = int(perms[m][i])
            record[f"Delta_cls_{m}"] = float(full_z[i, cls[i]]-z[i, cls[i]])
            record[f"Delta_reg_{m}"] = abs(float(full_y[i]-y[i]))
        for target in ("cls", "reg"):
            record[f"dominant_modality_{target}"] = max(ru.MODS, key=lambda m: record[f"Delta_{target}_{m}"])
        joint_z, joint_y = predictions["audio_vision"]
        record["Delta_cls_audio_vision_joint"] = float(full_z[i, cls[i]]-joint_z[i, cls[i]])
        record["Delta_reg_audio_vision_joint"] = abs(float(full_y[i]-joint_y[i]))
        rows.append(record)
    write_csv(OUT / "q3_modality_contribution.csv", rows)
    print("MODALITY_DONORS_PASS", flush=True)
    grids = candidate_grid(data, adapter)
    local = local_evidence(grids, data, text, adapter, model, full_z, full_y)
    faithful = faithfulness(grids, data, text, adapter, model, full_z, full_y)
    stable = stability(grids, data, text, adapter, model, full_z, full_y)
    text_pass = any(faithful["text"][t].get("passes_operator_consistency", False) for t in ("cls", "reg"))
    av_pass = any(faithful[m][t].get("passes_operator_consistency", False)
                  for m in ("audio", "vision") for t in ("cls", "reg"))
    status = "Q3_EXPLANATION_METHOD_ACCEPTED" if text_pass and av_pass else "Q3_EXPLANATION_METHOD_NOT_FAITHFUL"
    payload = {"status": status, "scope": "A2 unaligned valid 728 only", "seed": SEED,
               "checkpoint_sha256": file_sha256(RUN / "checkpoint.pt"),
               "method_sha256": file_sha256(OUT / "q3_explanation_method.md"),
               "donor_manifest_sha256": manifest_hash,
               "modality_contribution_csv_sha256": file_sha256(OUT / "q3_modality_contribution.csv"),
               "full_prediction_frozen_bridge_equivalent": True,
               "donor_aggregate": donor_summary, "local_evidence": local,
               "faithfulness": faithful, "stability": stable,
               "acceptance": {"text_pass_at_least_one_target": text_pass,
                              "audio_or_vision_pass_at_least_one_target": av_pass},
               "runtime_seconds": time.perf_counter()-start,
               "not_causal": True, "no_special_or_test_access": True}
    (OUT / "q3_faithfulness.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report(payload)
    print(json.dumps({"status": status, "seconds": payload["runtime_seconds"],
                      "text_pass": text_pass, "av_pass": av_pass}), flush=True)


if __name__ == "__main__":
    main()
