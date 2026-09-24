"""
Stage 3C pilot runner for E0 (independent H9 test).

Reuses pilot_stage4.py for data loading, clean/gap BERT caches, mask manifests,
augmented input construction, metrics, selection loss, and prediction writing.
Only the model (E0Head) differs from R1, and within it only temporal pooling.
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
from stage3c_models import E0Head

LAMBDA_REG = 2.0526115894317627
EPOCH_CAP = 12
PATIENCE = 3
RUN_ROOT = Path("workspace/experiments/stage3c_runs")
MODS = ("text", "audio", "vision")
MCODE = {0: "T", 1: "A", 2: "V"}


def local_diag(local, data, specs):
    """Aggregate local-weight mechanism for one chunk (a gap scenario)."""
    rows = {}
    for mi, m in enumerate(MODS):
        w = local[mi]["weights"].numpy()
        ent = -(w * np.log(np.clip(w, 1e-12, 1))).sum(axis=1)
        w_gap, w_bd, w_un, pos = [], [], [], []
        for k, (r, e) in enumerate(specs):
            pos.append(w[k])
            if e is not None and e["injectable"] and MCODE[mi] in e["spans"]:
                aa, bb = e["spans"][MCODE[mi]]
                w_gap.append(float(w[k, aa:bb].mean()))
                bpos = list(range(max(0, aa - 2), aa)) + list(range(bb, min(50, bb + 2)))
                if bpos:
                    w_bd.append(float(w[k, bpos].mean()))
            un = np.flatnonzero((data[f"{m}_O"][r] == 2) | (data[f"{m}_P"][r] == 2))
            if len(un):
                w_un.append(float(w[k, un].mean()))
        rows[m] = {
            "entropy": float(ent.mean()),
            "w_gap": float(np.mean(w_gap)) if w_gap else None,
            "w_boundary": float(np.mean(w_bd)) if w_bd else None,
            "w_unknown": float(np.mean(w_un)) if w_un else None,
            "position_mean": np.mean(np.stack(pos), 0).tolist()}
    return rows


def merge_diag(blocks):
    out = {}
    for m in MODS:
        bs = [b[m] for b in blocks if m in b]
        if not bs:
            continue
        out[m] = {
            k: (float(np.mean([b[k] for b in bs if b[k] is not None]))
                if k != "position_mean"
                else np.mean([b["position_mean"] for b in bs], 0).tolist())
            for k in ("entropy", "w_gap", "w_boundary", "w_unknown", "position_mean")}
    return out


def forward_chunks(model, data, clean_states, gap_states, lookup, entries=None):
    model.eval()
    zs, ys, dg = [], [], []
    n = len(data["samples"])
    for start in range(0, n, ps.BATCH_HEAD):
        rows = list(range(start, min(n, start + ps.BATCH_HEAD)))
        specs = [(i, entries[i] if entries is not None else None) for i in rows]
        inp = ps.augmented_inputs(data, clean_states, gap_states, lookup, specs)
        with torch.inference_mode():
            z, y, local = model(*inp)
        zs.append(z.numpy())
        ys.append(y.numpy())
        if entries is not None:
            dg.append(local_diag(local, data, specs))
    return np.concatenate(zs), np.concatenate(ys), (merge_diag(dg) if dg else None)


def evaluate(model, valid, valid_clean, gap_states, lookup, phase_entries):
    clean_z, clean_y, _ = forward_chunks(model, valid, valid_clean, gap_states, lookup, None)
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    out = {"clean": ps.metrics_from_arrays(yc, yr, clean_z, clean_y), "scenarios": {}}
    preds = {"clean": (clean_z, clean_y, None)}
    mech = {}
    for sc, entries in phase_entries.items():
        z, y, dg = forward_chunks(model, valid, valid_clean, gap_states, lookup, entries)
        out["scenarios"][sc] = ps.scenario_metrics(valid, clean_z, clean_y, z, y, entries)
        preds[sc] = (z, y, None)
        mech[sc] = dg
    return out, preds, mech


def run(seed):
    ps.set_determinism(seed)
    train, valid = ps.load_split("train"), ps.load_split("valid")
    train_entries, valid_entries = ps.masks_by_phase(seed)
    gap_meta = ps.cache_gap_bert(train, valid, seed)
    lookup = gap_meta["lookup"]
    train_clean = np.load(ps.CACHE / "clean_train.npy", mmap_mode="r")
    valid_clean = np.load(ps.CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(ps.CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")

    run_dir = RUN_ROOT / f"E0_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"name": "E0", "seed": seed, "lambda_reg": LAMBDA_REG, "epochs": EPOCH_CAP,
         "patience": PATIENCE, "lr": 0.001, "weight_decay": 0.0001,
         "data": "workspace/data/processed/cleaning_2026e_v2/aligned",
         "only_delta_vs_R1": "content temporal mean pooling -> local scorer weighted pooling"},
        indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    peak = ps.current_rss()
    model = E0Head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n = len(train["samples"])
    specs = [(i, None) for i in range(n)] + [(i, train_entries[i]) for i in range(n)]
    best_loss, best_epoch, no_improve, epoch_times = float("inf"), 0, 0, []
    for epoch in range(1, EPOCH_CAP + 1):
        et = time.perf_counter()
        model.train()
        order = np.random.default_rng(ps.stable_seed("shuffle", "E0", seed, epoch)).permutation(
            len(specs))
        accum = np.zeros(3)
        for start in range(0, len(order), ps.BATCH_HEAD):
            chosen = [specs[int(i)] for i in order[start:start + ps.BATCH_HEAD]]
            inp = ps.augmented_inputs(train, train_clean, gap_states, lookup, chosen)
            rows = np.array([i for i, _ in chosen])
            cy, ry = ps.targets(train, rows)
            optimizer.zero_grad(set_to_none=True)
            z, y, _ = model(*inp)
            ce = torch.nn.functional.cross_entropy(z, cy)
            huber = torch.nn.functional.huber_loss(y, ry, delta=1.0)
            loss = ce + LAMBDA_REG * huber
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            optimizer.step()
            accum += np.array([float(loss.detach()), float(ce.detach()),
                               float(huber.detach())]) * len(chosen)
        selection, pred = evaluate(model, valid, valid_clean, gap_states, lookup,
                                   valid_entries["selection"])[:2]
        score, _ = ps.selection_loss(pred, valid, LAMBDA_REG)
        epoch_sec = time.perf_counter() - et
        epoch_times.append(epoch_sec)
        peak = max(peak, ps.current_rss())
        ps.append_jsonl(run_dir / "training_log.jsonl",
                        {"epoch": epoch, "train_loss": (accum / len(specs)).tolist(),
                         "valid_selection_loss": score,
                         "clean_f1": selection["clean"]["macro_f1"],
                         "clean_mae": selection["clean"]["mae"],
                         "epoch_seconds": epoch_sec})
        print(json.dumps({"run": "E0", "epoch": epoch, "seconds": round(epoch_sec, 1),
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
    audit, predictions, mech = evaluate(model, valid, valid_clean, gap_states, lookup,
                                        valid_entries["audit"])
    ps.write_all_predictions(run_dir, valid, predictions)
    result = {"status": "COMPLETE", "name": "E0", "seed": seed, "lambda_reg": LAMBDA_REG,
              "audit": audit, "best_epoch": best_epoch,
              "best_valid_selection_loss": best_loss,
              "mean_epoch_seconds": float(np.mean(epoch_times)),
              "train_wall_seconds": time.perf_counter() - t0,
              "peak_rss_bytes": peak,
              "checkpoint_sha256": ps.sha256(run_dir / "checkpoint.pt"),
              "local_mechanism": mech}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("CLEAN", {k: round(audit["clean"][k], 4) for k in
                    ("accuracy", "macro_f1", "mae", "pearson")})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    run(args.seed)
