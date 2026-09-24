"""Q2 Stage 3D E1 pilot; train/valid only, frozen Stage 4 comparators."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import pilot_stage4 as ps
from stage3d_models import E1Head, global_loss, local_loss

HERE = Path(__file__).resolve().parent
PLAN = HERE / "q2_stage3d_plan.md"
RESULT = Path("workspace/results/q2_stage3d")
RUN_ROOT = HERE / "stage3d_runs"
LAMBDA_REG = 2.0526115894317627
MODS = ("text", "audio", "vision")
CODES = ("T", "A", "V")


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def frozen_check() -> dict:
    RESULT.mkdir(parents=True, exist_ok=True)
    assert PLAN.exists()
    assert ps.sha256(ps.ROOT / "manifest.json")
    manifest = json.loads((ps.ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract_sha256"] == ps.EXPECTED["contract_sha256"]
    assert ps.sha256(ps.ALIGNED / "scalers.json") == ps.EXPECTED["scaler_sha256"]
    assert ps.sha256(ps.BERT / "pytorch_model.bin") == ps.EXPECTED["bert_weights_sha256"]
    assert ps.sha256(ps.BERT / "vocab.txt") == ps.EXPECTED["bert_vocab_sha256"]
    assert len(ps.load_split("train")["samples"]) == 3395
    assert len(ps.load_split("valid")["samples"]) == 728
    masks = {}
    comparator = {}
    for seed in ps.SEEDS:
        mpath = ps.mask_path(seed)
        assert mpath.exists()
        mask_sha = ps.sha256(mpath)
        masks[str(seed)] = mask_sha
        comparator[str(seed)] = {}
        for name, folder in (("R0a", ps.RUN_DIR / f"R0a_seed{seed}"),
                             ("R1", ps.RUN_DIR / f"R1_seed{seed}"),
                             ("E0", HERE / "stage3c_runs" / f"E0_seed{seed}")):
            metrics = folder / "metrics.json"
            config = folder / "config.json"
            assert metrics.exists() and config.exists(), f"missing frozen {name} seed {seed}"
            obj = json.loads(metrics.read_text(encoding="utf-8"))
            assert obj["status"] == "COMPLETE"
            if name != "E0":
                assert json.loads(config.read_text(encoding="utf-8"))["mask_sha256"] == mask_sha
            comparator[str(seed)][name] = {"metrics_sha256": ps.sha256(metrics),
                                            "config_sha256": ps.sha256(config)}
    record = {"status": "PASSED", "plan_sha256": ps.sha256(PLAN),
              "model_code_sha256": ps.sha256(HERE / "stage3d_models.py"),
              "pilot_code_sha256": ps.sha256(Path(__file__)),
              "v2_manifest_sha256": ps.sha256(ps.ROOT / "manifest.json"),
              "scaler_sha256": ps.EXPECTED["scaler_sha256"],
              "bert_weights_sha256": ps.EXPECTED["bert_weights_sha256"],
              "mask_sha256": masks, "comparators": comparator,
              "allowed_splits": ["train", "valid"]}
    write(RESULT / "frozen_preflight.json", record)
    return record


def inputs_and_mask(data: dict, clean: np.ndarray, gap: np.ndarray, lookup: dict,
                    specs: list[tuple[int, dict]]):
    inp = ps.augmented_inputs(data, clean, gap, lookup, specs)
    mask = np.zeros((len(specs), 3, 50), dtype=bool)
    for i, (row, entry) in enumerate(specs):
        if entry is None or not entry["injectable"]:
            continue
        for m, code in enumerate(CODES):
            span = entry["spans"].get(code)
            if span is not None:
                a, b = span
                assert np.all(data[f"{MODS[m]}_P"][row, a:b] == 1)
                assert np.all(data[f"{MODS[m]}_O"][row, a:b] == 1)
                mask[i, m, a:b] = True
    out = torch.from_numpy(mask)
    assert not torch.any(out & ~((inp[3] == 1) & (inp[4] == 0)))
    return inp, out


def clean_pairs(data: dict, clean: np.ndarray, gap: np.ndarray, lookup: dict,
                specs: list[tuple[int, dict]]):
    return ps.augmented_inputs(data, clean, gap, lookup, [(i, None) for i, _ in specs])


def loaded(seed: int):
    train, valid = ps.load_split("train"), ps.load_split("valid")
    train_entries, valid_entries = ps.masks_by_phase(seed)
    gap_meta = ps.cache_gap_bert(train, valid, seed)
    assert gap_meta["mask_sha256"] == ps.sha256(ps.mask_path(seed))
    clean_train = np.load(ps.CACHE / "clean_train.npy", mmap_mode="r")
    clean_valid = np.load(ps.CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(ps.CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")
    assert clean_train.shape == (3395, 50, 768)
    assert clean_valid.shape == (728, 50, 768)
    return train, valid, train_entries, valid_entries, gap_meta, clean_train, clean_valid, gap_states


def calibration() -> dict:
    path = RESULT / "lambda_calibration.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert obj["plan_sha256"] == ps.sha256(PLAN)
        return obj
    frozen_check()
    ps.set_determinism(1729)
    train, _, train_entries, _, gap_meta, clean_train, _, gap_states = loaded(1729)
    specs = [(i, train_entries[i]) for i in range(3395) if train_entries[i]["injectable"]][:256]
    assert len(specs) == 256
    ps.set_determinism(1729)
    model = E1Head().eval()
    totals = np.zeros(3, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, 256, 64):
            batch = specs[start:start+64]
            gap_in, inj = inputs_and_mask(train, clean_train, gap_states, gap_meta["lookup"], batch)
            full_in = clean_pairs(train, clean_train, gap_states, gap_meta["lookup"], batch)
            gap_out = model(*gap_in, injected=inj, diagnostics=True)
            full_out = model(*full_in, diagnostics=True)
            cy, ry = ps.targets(train, np.array([i for i, _ in batch]))
            pred, _, _ = ps.losses(gap_out["logits"], gap_out["intensity"], cy, ry, LAMBDA_REG)
            loc = local_loss(gap_out, full_out, inj)
            glob = global_loss(gap_out, full_out, inj)
            totals += np.array([float(pred), float(loc), float(glob)]) * len(batch)
    means = totals / 256
    assert np.isfinite(means).all() and means[1] > 1e-8 and means[2] > 1e-8
    lam_local = float(np.clip(0.1 * means[0] / means[1], 0.01, 10.0))
    lam_global = float(np.clip(0.1 * means[0] / means[2], 0.01, 10.0))
    record = {"status": "FROZEN_TRAIN_ONLY", "plan_sha256": ps.sha256(PLAN),
              "seed": 1729, "sample_count": 256,
              "sample_id_sha256": __import__("hashlib").sha256("\n".join(train["samples"][i]["id"] for i, _ in specs).encode()).hexdigest(),
              "train_loss_means": {"prediction": float(means[0]), "local": float(means[1]),
                                   "global": float(means[2])},
              "target_auxiliary_fraction_each": 0.10,
              "clip_interval": [0.01, 10.0],
              "lambda_local": lam_local, "lambda_global": lam_global,
              "lambda_reg": LAMBDA_REG,
              "mask_sha256": ps.sha256(ps.mask_path(1729)),
              "scaler_sha256": ps.EXPECTED["scaler_sha256"],
              "bert_weights_sha256": ps.EXPECTED["bert_weights_sha256"],
              "valid_predictions_inspected": False}
    write(path, record)
    print(json.dumps({"calibration": record["train_loss_means"],
                      "lambda_local": lam_local, "lambda_global": lam_global}), flush=True)
    return record


def predict_specs(model, data, clean, gap, lookup, entries, restore_enabled):
    model.eval()
    logits, intensity = [], []
    with torch.inference_mode():
        for start in range(0, len(data["samples"]), ps.BATCH_HEAD):
            specs = [(i, entries[i] if entries is not None else None)
                     for i in range(start, min(start+ps.BATCH_HEAD, len(data["samples"]))) ]
            inp, inj = inputs_and_mask(data, clean, gap, lookup, specs)
            out = model(*inp, injected=inj, restoration_enabled=restore_enabled)
            logits.append(out["logits"].numpy())
            intensity.append(out["intensity"].numpy())
    return np.concatenate(logits), np.concatenate(intensity), None


def evaluate(model, valid, clean, gap, lookup, phase_entries, restore_enabled):
    clean_pred = predict_specs(model, valid, clean, gap, lookup, None, restore_enabled)
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    result = {"clean": ps.metrics_from_arrays(yc, yr, clean_pred[0], clean_pred[1]), "scenarios": {}}
    preds = {"clean": clean_pred}
    for sc, entries in phase_entries.items():
        pred = predict_specs(model, valid, clean, gap, lookup, entries, restore_enabled)
        result["scenarios"][sc] = ps.scenario_metrics(valid, clean_pred[0], clean_pred[1],
                                                       pred[0], pred[1], entries)
        preds[sc] = pred
    return result, preds


def mechanism(model, valid, clean, gap, lookup, entries_by_scenario, restore_enabled):
    model.eval()
    by_sc = {}
    with torch.inference_mode():
        for sc, entries in entries_by_scenario.items():
            local_restored = local_zero = dist_restored = dist_untreated = 0.0
            n_positions = n_samples = n_native_unknown = 0
            for start in range(0, 728, ps.BATCH_HEAD):
                specs = [(i, entries[i]) for i in range(start, min(728, start+ps.BATCH_HEAD))
                         if entries[i]["injectable"]]
                if not specs:
                    continue
                gap_in, inj = inputs_and_mask(valid, clean, gap, lookup, specs)
                full_in = clean_pairs(valid, clean, gap, lookup, specs)
                gap_out = model(*gap_in, injected=inj,
                                restoration_enabled=restore_enabled, diagnostics=True)
                full_out = model(*full_in, diagnostics=True)
                n_native_unknown += int(((gap_in[4] == 2) & ~inj).sum())
                n_samples += len(specs)
                dist_restored += float((1 - F.cosine_similarity(gap_out["H"], full_out["H"], dim=-1)).sum())
                dist_untreated += float((1 - F.cosine_similarity(gap_out["H_untreated"], full_out["H"], dim=-1)).sum())
                for m in range(3):
                    mask = inj[:, m]
                    if not torch.any(mask):
                        continue
                    target = full_out["z"][m][mask]
                    n_positions += int(mask.sum())
                    local_zero += float(F.huber_loss(torch.zeros_like(target), target,
                                                     delta=1.0, reduction="none").mean(dim=-1).sum())
                    if gap_out["z_hat"][m] is not None:
                        local_restored += float(F.huber_loss(gap_out["z_hat"][m][mask], target,
                                                             delta=1.0, reduction="none").mean(dim=-1).sum())
            by_sc[sc] = {"n_samples": n_samples, "n_injected_positions": n_positions,
                         "native_unknown_positions_not_restored": n_native_unknown,
                         "local_restored_huber": local_restored/max(1,n_positions) if restore_enabled else None,
                         "local_zero_huber": local_zero/max(1,n_positions),
                         "distance_after": dist_restored/max(1,n_samples),
                         "distance_untreated": dist_untreated/max(1,n_samples)}
    mean_local = float(np.mean([v["local_restored_huber"] for v in by_sc.values()])) if restore_enabled else None
    mean_zero = float(np.mean([v["local_zero_huber"] for v in by_sc.values()]))
    mean_after = float(np.mean([v["distance_after"] for v in by_sc.values()]))
    mean_untreated = float(np.mean([v["distance_untreated"] for v in by_sc.values()]))
    gain_local = (1 - mean_local/mean_zero) if restore_enabled and mean_zero > 1e-8 else None
    gain_global = (1 - mean_after/mean_untreated) if mean_untreated > 1e-8 else None
    return {"scenarios": by_sc, "six_scenario_means": {
        "local_restored_huber": mean_local, "local_zero_huber": mean_zero,
        "distance_after": mean_after, "distance_untreated": mean_untreated,
        "restoration_gain": gain_local, "representation_gain": gain_global},
        "mechanism_pass": bool(gain_local is not None and gain_global is not None
                               and gain_local >= 0.10 and gain_global >= 0.10)}


def fit(seed: int, mode: str) -> dict:
    if mode != "full" and seed != 1729:
        raise RuntimeError("ablations permitted only for seed1729")
    cal = calibration()
    pre = json.loads((RESULT / "frozen_preflight.json").read_text(encoding="utf-8"))
    assert pre["plan_sha256"] == ps.sha256(PLAN)
    ps.set_determinism(seed)
    train, valid, train_entries, valid_entries, gap_meta, train_clean, valid_clean, gap_states = loaded(seed)
    run_id = f"E1_{mode}_seed{seed}"
    run = RUN_ROOT / run_id
    run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ps.mask_path(seed), run / "mask_manifest.jsonl")
    restore_enabled = mode != "consistency_only"
    lam_local = cal["lambda_local"] if restore_enabled else 0.0
    lam_global = cal["lambda_global"] if mode != "local_only" else 0.0
    config = {"run_id": run_id, "seed": seed, "mode": mode,
              "source_version": "cleaning_2026e_v2/aligned", "allowed_splits": ["train", "valid"],
              "epochs_cap": 12, "patience": 3, "batch_size": ps.BATCH_HEAD,
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "lambda_reg": LAMBDA_REG, "lambda_local": lam_local,
              "lambda_global": lam_global, "restoration_enabled": restore_enabled,
              "mask_sha256": ps.sha256(run / "mask_manifest.jsonl"),
              "plan_sha256": ps.sha256(PLAN), "calibration_sha256": ps.sha256(RESULT / "lambda_calibration.json")}
    assert config["mask_sha256"] == pre["mask_sha256"][str(seed)]
    write(run / "config.json", config)
    write(run / "environment_checksums.json", {"torch": torch.__version__,
          "python": sys.version, "cuda": torch.cuda.is_available(),
          "model_code_sha256": ps.sha256(HERE / "stage3d_models.py"),
          "runner_sha256": ps.sha256(Path(__file__)), "plan_sha256": ps.sha256(PLAN),
          "v2_manifest_sha256": pre["v2_manifest_sha256"],
          "scaler_sha256": pre["scaler_sha256"],
          "bert_weights_sha256": pre["bert_weights_sha256"],
          "mask_sha256": config["mask_sha256"], "frozen_comparators": pre["comparators"][str(seed)]})
    for name in ("training_log.jsonl", "predictions_valid.jsonl", "failures.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    ps.set_determinism(seed)
    model = E1Head()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    specs = [(i, None) for i in range(3395)] + [(i, train_entries[i]) for i in range(3395)]
    best, best_epoch, stale = float("inf"), 0, 0
    epoch_times = []
    start_run = time.perf_counter()
    peak = ps.current_rss()
    try:
        for epoch in range(1, 13):
            t0 = time.perf_counter()
            model.train()
            order = np.random.default_rng(ps.stable_seed("shuffle", run_id, seed, epoch)).permutation(len(specs))
            sums = np.zeros(5, dtype=np.float64)
            for begin in range(0, len(specs), ps.BATCH_HEAD):
                batch = [specs[int(i)] for i in order[begin:begin+ps.BATCH_HEAD]]
                gap_in, inj = inputs_and_mask(train, train_clean, gap_states, gap_meta["lookup"], batch)
                rows = np.array([i for i, _ in batch])
                cy, ry = ps.targets(train, rows)
                opt.zero_grad(set_to_none=True)
                out = model(*gap_in, injected=inj,
                            restoration_enabled=restore_enabled, diagnostics=bool(torch.any(inj)))
                pred, ce, huber = ps.losses(out["logits"], out["intensity"], cy, ry, LAMBDA_REG)
                loc = pred * 0.0
                glob = pred * 0.0
                if torch.any(inj) and (lam_local or lam_global):
                    full_in = clean_pairs(train, train_clean, gap_states, gap_meta["lookup"], batch)
                    with torch.no_grad():
                        full = model(*full_in, diagnostics=True)
                    if lam_local:
                        loc = local_loss(out, full, inj)
                    if lam_global:
                        glob = global_loss(out, full, inj)
                loss = pred + lam_local * loc + lam_global * glob
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite E1 loss")
                loss.backward()
                opt.step()
                sums += np.array([float(loss.detach()), float(ce.detach()),
                                  float(huber.detach()), float(loc.detach()),
                                  float(glob.detach())]) * len(batch)
            selection, preds = evaluate(model, valid, valid_clean, gap_states,
                                        gap_meta["lookup"], valid_entries["selection"], restore_enabled)
            score, detail = ps.selection_loss(preds, valid, LAMBDA_REG)
            elapsed = time.perf_counter() - t0
            epoch_times.append(elapsed)
            peak = max(peak, ps.current_rss())
            ps.append_jsonl(run / "training_log.jsonl", {"epoch": epoch,
                "train_loss_total_ce_huber_local_global": (sums/len(specs)).tolist(),
                "valid_selection_loss": score, "valid_loss_detail": detail,
                "valid_selection_clean": selection["clean"],
                "epoch_seconds": elapsed, "peak_rss_bytes": peak})
            print(json.dumps({"run": run_id, "epoch": epoch, "seconds": round(elapsed, 2),
                              "selection_loss": round(score, 5),
                              "clean_f1": round(selection["clean"]["macro_f1"], 4),
                              "clean_mae": round(selection["clean"]["mae"], 4)}), flush=True)
            if score < best - 1e-6:
                best, best_epoch, stale = score, epoch, 0
                torch.save(model.state_dict(), run / "checkpoint.pt")
            else:
                stale += 1
            if peak > ps.MAX_RSS_BYTES or (time.perf_counter()-start_run + max(epoch_times)*(12-epoch)) > ps.MAX_SEED_SECONDS:
                raise RuntimeError("Q2_STAGE3D_BLOCKED: RAM/time cap")
            if stale >= 3:
                break
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        audit, preds = evaluate(model, valid, valid_clean, gap_states,
                                gap_meta["lookup"], valid_entries["audit"], restore_enabled)
        ps.write_all_predictions(run, valid, preds)
        mech = mechanism(model, valid, valid_clean, gap_states, gap_meta["lookup"],
                         valid_entries["audit"], restore_enabled)
        result = {"status": "COMPLETE", "run_id": run_id, "mode": mode,
                  "seed": seed, "audit": audit, "mechanism": mech,
                  "best_epoch": best_epoch, "best_valid_selection_loss": best,
                  "epoch_times_seconds": epoch_times,
                  "train_wall_seconds": time.perf_counter()-start_run,
                  "peak_rss_bytes": peak,
                  "checkpoint_sha256": ps.sha256(run / "checkpoint.pt"),
                  "mask_sha256": config["mask_sha256"]}
        write(run / "metrics.json", result)
        return result
    except Exception as exc:
        failure = {"status": "FAILED", "run_id": run_id, "error": repr(exc),
                   "elapsed_seconds": time.perf_counter()-start_run,
                   "completed_epochs": len(epoch_times), "peak_rss_bytes": peak}
        ps.append_jsonl(run / "failures.jsonl", failure)
        write(run / "metrics.json", failure)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["preflight", "calibrate", "run"])
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--mode", choices=["full", "local_only", "consistency_only"], default="full")
    args = ap.parse_args()
    if args.command == "preflight":
        print(json.dumps(frozen_check(), ensure_ascii=False))
    elif args.command == "calibrate":
        calibration()
    else:
        assert args.seed in ps.SEEDS
        result = fit(args.seed, args.mode)
        print(json.dumps({"run": result["run_id"], "clean": result["audit"]["clean"],
                          "mechanism": result["mechanism"]["six_scenario_means"]}), flush=True)


if __name__ == "__main__":
    main()
