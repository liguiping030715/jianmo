"""
Stage 3B pilot runner (D0 / D1).

Reuses the validated Stage-4 pipeline in pilot_stage4.py for:
  data loading (aligned v2), clean/gap BERT caches, mask manifests,
  augmented input construction, metrics, selection loss, predictions.
Only the model forward and loss wrapper differ, so the ablations are isolated:
  frozen R1 -> D0 : shared fusion -> task-specific fusion (H8)
  D0 -> D1       : mean pooling -> local state-aware pooling (H9)
"""
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_stage4 as ps
from stage3b_models import (D0TaskSpecificFusion, D1LocalStateTemporalPooling,
                            count_trainable)

LAMBDA_REG = 2.0526115894317627
EPOCH_CAP = 12
PATIENCE = 3
RUN_ROOT = Path("workspace/experiments/stage3b_runs")
MODS = ("text", "audio", "vision")
MCODE = {0: "T", 1: "A", 2: "V"}


def build_model(name):
    if name == "D0":
        return D0TaskSpecificFusion(hidden=128, max_len=50)
    if name == "D1":
        return D1LocalStateTemporalPooling(hidden=128, scorer_hidden=32,
                                           max_len=50, kernel_size=3)
    raise ValueError(name)


def to_batch(inp):
    t, a, v, p, o = inp
    p = p.long()
    o = o.long()
    return {"text_states": t, "audio": a, "vision": v,
            "p_text": p[:, 0], "o_text": o[:, 0],
            "p_audio": p[:, 1], "o_audio": o[:, 1],
            "p_vision": p[:, 2], "o_vision": o[:, 2]}


def forward_chunks(model, data, clean_states, gap_states, lookup, entries=None,
                   batch_size=ps.BATCH_HEAD):
    model.eval()
    zs, ys, dg = [], [], []
    n = len(data["samples"])
    for start in range(0, n, batch_size):
        rows = list(range(start, min(n, start + batch_size)))
        specs = [(i, entries[i] if entries is not None else None) for i in rows]
        inp = ps.augmented_inputs(data, clean_states, gap_states, lookup, specs)
        with torch.inference_mode():
            out = model(to_batch(inp))
        zs.append(out["logits"].numpy())
        ys.append(out["intensity"].numpy())
        if "D1" in model.__class__.__name__ and entries is not None:
            dg.append(local_diag(out, data, specs))
    z = np.concatenate(zs)
    y = np.concatenate(ys)
    return z, y, (merge_diag(dg) if dg else None)


def local_diag(out, data, specs):
    rows = {}
    for mi, m in enumerate(MODS):
        d = out["diagnostics"][m]
        if "local_weights" not in d:
            continue
        w = d["local_weights"].numpy()
        ent = -(w * np.log(np.clip(w, 1e-12, 1))).sum(axis=1)
        w_in, w_bd, w_un, pos = [], [], [], []
        for k, (r, e) in enumerate(specs):
            pos.append(w[k])
            if e is not None and e["injectable"] and MCODE[mi] in e["spans"]:
                aa, bb = e["spans"][MCODE[mi]]
                w_in.append(float(w[k, aa:bb].mean()))
                bpos = list(range(max(0, aa - 2), aa)) + list(range(bb, min(50, bb + 2)))
                if bpos:
                    w_bd.append(float(w[k, bpos].mean()))
            un = np.flatnonzero((data[f"{m}_O"][r] == 2) | (data[f"{m}_P"][r] == 2))
            if len(un):
                w_un.append(float(w[k, un].mean()))
        rows[m] = {"entropy": float(ent.mean()),
                   "w_gap": float(np.mean(w_in)) if w_in else None,
                   "w_boundary": float(np.mean(w_bd)) if w_bd else None,
                   "w_unknown": float(np.mean(w_un)) if w_un else None,
                   "position_mean": np.mean(np.stack(pos), 0).tolist()}
    return rows


def merge_diag(dg):
    out = {}
    for m in MODS:
        blocks = [d[m] for d in dg if m in d]
        if not blocks:
            continue
        out[m] = {
            k: (float(np.mean([b[k] for b in blocks if b[k] is not None]))
                if k != "position_mean"
                else np.mean([b["position_mean"] for b in blocks], 0).tolist())
            for k in ("entropy", "w_gap", "w_boundary", "w_unknown", "position_mean")}
    return out


def evaluate(model, valid, valid_clean, gap_states, lookup, phase_entries):
    clean_z, clean_y, _ = forward_chunks(model, valid, valid_clean, gap_states, lookup, None)
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    out = {"clean": ps.metrics_from_arrays(yc, yr, clean_z, clean_y), "scenarios": {}}
    preds = {"clean": (clean_z, clean_y, None)}
    local = {}
    for sc, entries in phase_entries.items():
        z, y, dg = forward_chunks(model, valid, valid_clean, gap_states, lookup, entries)
        out["scenarios"][sc] = ps.scenario_metrics(valid, clean_z, clean_y, z, y, entries)
        preds[sc] = (z, y, None)
        if dg:
            local[sc] = dg
    return out, preds, local


def run(name, seed):
    ps.set_determinism(seed)
    train, valid = ps.load_split("train"), ps.load_split("valid")
    train_entries, valid_entries = ps.masks_by_phase(seed)
    gap_meta = ps.cache_gap_bert(train, valid, seed)
    lookup = gap_meta["lookup"]
    train_clean = np.load(ps.CACHE / "clean_train.npy", mmap_mode="r")
    valid_clean = np.load(ps.CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(ps.CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")

    run_dir = RUN_ROOT / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"name": name, "seed": seed, "lambda_reg": LAMBDA_REG,
         "epochs": EPOCH_CAP, "patience": PATIENCE, "lr": 0.001, "weight_decay": 0.0001,
         "data": "workspace/data/processed/cleaning_2026e_v2/aligned"}, indent=2),
        encoding="utf-8")

    t0 = time.perf_counter()
    peak = ps.current_rss()
    model = build_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n = len(train["samples"])
    specs = [(i, None) for i in range(n)] + [(i, train_entries[i]) for i in range(n)]
    best_loss, best_epoch, no_improve, epoch_times = float("inf"), 0, 0, []
    for epoch in range(1, EPOCH_CAP + 1):
        et = time.perf_counter()
        model.train()
        order = np.random.default_rng(ps.stable_seed("shuffle", name, seed, epoch)).permutation(
            len(specs))
        accum = np.zeros(3)
        for start in range(0, len(order), ps.BATCH_HEAD):
            chosen = [specs[int(i)] for i in order[start:start + ps.BATCH_HEAD]]
            inp = ps.augmented_inputs(train, train_clean, gap_states, lookup, chosen)
            rows = np.array([i for i, _ in chosen])
            cy, ry = ps.targets(train, rows)
            optimizer.zero_grad(set_to_none=True)
            out = model(to_batch(inp))
            ce = torch.nn.functional.cross_entropy(out["logits"], cy)
            huber = torch.nn.functional.huber_loss(out["intensity"], ry, delta=1.0)
            loss = ce + LAMBDA_REG * huber
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            optimizer.step()
            accum += np.array([float(loss.detach()), float(ce.detach()),
                               float(huber.detach())]) * len(chosen)
        selection, pred = evaluate(model, valid, valid_clean, gap_states, lookup,
                                   valid_entries["selection"])[:2]
        score, detail = ps.selection_loss(pred, valid, LAMBDA_REG)
        epoch_sec = time.perf_counter() - et
        epoch_times.append(epoch_sec)
        peak = max(peak, ps.current_rss())
        ps.append_jsonl(run_dir / "training_log.jsonl",
                        {"epoch": epoch, "train_loss": (accum / len(specs)).tolist(),
                         "valid_selection_loss": score,
                         "clean_f1": selection["clean"]["macro_f1"],
                         "clean_mae": selection["clean"]["mae"],
                         "epoch_seconds": epoch_sec})
        print(json.dumps({"run": name, "epoch": epoch, "seconds": round(epoch_sec, 1),
                          "sel_loss": round(score, 4),
                          "f1": round(selection["clean"]["macro_f1"], 4),
                          "mae": round(selection["clean"]["mae"], 4)}, ensure_ascii=False),
              flush=True)
        if score < best_loss - 1e-6:
            best_loss, best_epoch, no_improve = score, epoch, 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            no_improve += 1
        if peak > ps.MAX_RSS_BYTES:
            raise RuntimeError("BUDGET_BLOCKED: RAM >14GiB")
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                                     weights_only=True))
    audit, predictions, local = evaluate(model, valid, valid_clean, gap_states, lookup,
                                         valid_entries["audit"])
    ps.write_all_predictions(run_dir, valid, predictions)
    result = {"status": "COMPLETE", "name": name, "seed": seed,
              "lambda_reg": LAMBDA_REG, "audit": audit,
              "best_epoch": best_epoch, "best_valid_selection_loss": best_loss,
              "mean_epoch_seconds": float(np.mean(epoch_times)),
              "train_wall_seconds": time.perf_counter() - t0,
              "peak_rss_bytes": peak,
              "checkpoint_sha256": ps.sha256(run_dir / "checkpoint.pt"),
              "trainable_params": count_trainable(model)}
    if local:
        result["local_weight_diagnostics"] = local
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("CLEAN", {k: round(audit["clean"][k], 4) for k in
                    ("accuracy", "macro_f1", "mae", "pearson")})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["D0", "D1"])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    run(args.model, args.seed)
