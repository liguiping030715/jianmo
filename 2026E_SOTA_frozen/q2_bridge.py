"""Frozen Q2 native-grid M0/M1 bridge pilot. See workspace/experiments/q2_bridge_plan.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

import pilot_ru as ru

BASE = Path("workspace/experiments/q2_bridge_runs")
CACHE = Path("workspace/experiments/pilot_cache")
PLAN = Path("workspace/experiments/q2_bridge_plan.md")
BATCH = 32
EPOCHS = 12
PATIENCE = 3
DIM = 48
HEADS = 2
FEED = 96
DROP = 0.1
LR = 0.001
WD = 0.0001
LAM = 2.0526115894317627
MAX_RSS = 14 * 1024 ** 3
MAX_SECONDS = 7200
SEEDS = (1729, 2718, 31415)
MODS = ru.MODS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sin_pos(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                    (-np.log(10000.0) / dim))
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(pos * div)
    out[:, 1::2] = torch.cos(pos * div)
    return out


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(DIM, HEADS, dropout=DROP, batch_first=True)
        self.norm1 = nn.LayerNorm(DIM)
        self.norm2 = nn.LayerNorm(DIM)
        self.ff = nn.Sequential(nn.Linear(DIM, FEED), nn.GELU(),
                                nn.Dropout(DROP), nn.Linear(FEED, DIM))
        self.dropout = nn.Dropout(DROP)

    def forward(self, query, key, key_unavailable):
        # A fully unavailable source has no softmax support. Unmask a neutral
        # zero key solely for numerical safety, then zero the attention output.
        all_bad = key_unavailable.all(dim=1)
        safe_mask = key_unavailable.clone()
        safe_key = key
        if bool(all_bad.any()):
            safe_mask[all_bad, 0] = False
            safe_key = key.clone()
            safe_key[all_bad, 0] = 0
        attended, _ = self.attn(query, safe_key, safe_key,
                                key_padding_mask=safe_mask, need_weights=False)
        attended = attended.masked_fill(all_bad[:, None, None], 0)
        x = self.norm1(query + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ff(x)))


class NativeMulT(nn.Module):
    def __init__(self, adapter: bool):
        super().__init__()
        self.adapter = adapter
        self.proj = nn.ModuleList([nn.Linear(ru.FEAT[m], DIM) for m in MODS])
        for m in MODS:
            self.register_buffer(f"pos_{m}", sin_pos(ru.LEN[m], DIM), persistent=False)
        self.audio_to_text = CrossBlock()
        self.vision_to_text = CrossBlock()
        self.fuse = nn.Sequential(nn.Linear(DIM * 3, DIM), nn.GELU(),
                                  nn.Dropout(DROP))
        self.cls = nn.Linear(DIM, 3)
        self.reg = nn.Linear(DIM, 1)
        # Construct after the common backbone so shared weights start identically
        # for a paired seed; only these learned state embeddings differ.
        if adapter:
            self.p_embed = nn.ModuleList([nn.Embedding(3, DIM) for _ in MODS])
            self.o_embed = nn.ModuleList([nn.Embedding(3, DIM) for _ in MODS])

    @staticmethod
    def pool(x, usable):
        w = usable.float().unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1)

    def forward(self, text, audio, vision, p_list, o_list, unknown_excluded=False):
        inputs = (text, audio, vision)
        features = []
        available = []
        for i, m in enumerate(MODS):
            x = self.proj[i](inputs[i]) + getattr(self, f"pos_{m}")[None]
            if self.adapter:
                x = x + self.p_embed[i](p_list[i].long()) + self.o_embed[i](o_list[i].long())
            features.append(F.gelu(x))
            available.append((p_list[i] != 0) &
                             ((o_list[i] != 0) if self.adapter else True))
        t, a, v = features
        ta = self.audio_to_text(t, a, ~available[1])
        tv = self.vision_to_text(t, v, ~available[2])
        summary = torch.cat([self.pool(x, available[0]) for x in (t, ta, tv)], 1)
        h = self.fuse(summary)
        return self.cls(h), 3.0 * torch.tanh(self.reg(h)).squeeze(1), None


def append(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def preflight():
    if not PLAN.exists():
        raise RuntimeError("frozen plan missing")
    if digest(ru.ROOT / "manifest.json") != "a1ae46325f3df842add1f5f384e16b5b0af92e487a909ee7bc67ac7d82440a83":
        raise RuntimeError("v2 manifest checksum mismatch")
    # RU's legacy EXPECTED scaler hash names the aligned scaler. The native
    # unaligned scaler has its own frozen v2 checksum.
    if digest(ru.UNALIGNED / "scalers.json") != "7c6a50015d705c9afd7b3c98e6a729c35a92a08fda619d3be632ebacaba8bb28":
        raise RuntimeError("scaler checksum mismatch")
    for name, expected in (("pytorch_model.bin", "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a"),
                           ("vocab.txt", ru.EXPECTED["bert_vocab_sha256"])):
        if digest(ru.BERT / name) != expected:
            raise RuntimeError(f"BERT {name} checksum mismatch")
    data = {s: ru.load_split(s) for s in ("train", "valid")}
    for split, d in data.items():
        if len(d["samples"]) != ru.SHAPES[split]:
            raise RuntimeError(f"{split} row count mismatch")
        for m in MODS:
            expected_shape = (ru.SHAPES[split], ru.LEN[m])
            if d[f"{m}_P"].shape != expected_shape or d[f"{m}_O"].shape != expected_shape:
                raise RuntimeError(f"{split}/{m} P/O shape mismatch")
            if not set(np.unique(d[f"{m}_P"])).issubset({0, 1, 2}) or not set(np.unique(d[f"{m}_O"])).issubset({0, 1, 2}):
                raise RuntimeError(f"{split}/{m} P/O domain mismatch")
            if m != "text" and d[m].shape != (*expected_shape, ru.FEAT[m]):
                raise RuntimeError(f"{split}/{m} feature shape mismatch")
        if d["text_bert"].shape != (ru.SHAPES[split], 3, 50):
            raise RuntimeError(f"{split} token shape mismatch")
        if len({x["id"] for x in d["samples"]}) != ru.SHAPES[split]:
            raise RuntimeError(f"{split} ID collision")
        token_hash = hashlib.sha256(d["text_bert"].tobytes()).hexdigest()
        if token_hash != json.loads((CACHE / "clean_text_cache.json").read_text(encoding="utf-8"))["tokens_sha256"][split]:
            raise RuntimeError(f"{split} BERT cache source mismatch")
        if np.load(CACHE / f"clean_{split}.npy", mmap_mode="r").shape != (ru.SHAPES[split], 50, 768):
            raise RuntimeError("clean BERT state shape mismatch")
    mask_hashes = {}
    for seed in SEEDS:
        path = ru.mask_path(seed)
        expected = ru.serialize_masks(ru.mask_rows(data["train"], data["valid"], seed))
        if path.read_bytes() != expected:
            raise RuntimeError(f"frozen native mask mismatch: {seed}")
        mask_hashes[str(seed)] = digest(path)
    # The old RU JSON contains a stale BERT-hash literal. Verify actual cached
    # states against the on-disk frozen encoder instead of trusting that label.
    from transformers import BertModel
    encoder = BertModel.from_pretrained(str(ru.BERT), local_files_only=True).eval()
    with torch.inference_mode():
        for split in ("train", "valid"):
            token = torch.from_numpy(data[split]["text_bert"][:16].copy())
            state = encoder(input_ids=token[:, 0], attention_mask=token[:, 1],
                            token_type_ids=token[:, 2]).last_hidden_state.numpy()
            saved = np.load(CACHE / f"clean_{split}.npy", mmap_mode="r")[:16]
            if not np.allclose(state, saved, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"{split} clean BERT cache content mismatch")
        for seed in SEEDS:
            rows = [json.loads(s) for s in ru.mask_path(seed).read_text(encoding="utf-8").splitlines()]
            entries = [x for x in rows if x["injectable"] and "text" in x["modalities"]][:16]
            tokens = []
            for entry in entries:
                token = data[entry["split"]]["text_bert"][entry["row"]].copy()
                lo, hi = entry["spans"]["text"]
                token[0, lo:hi] = 100
                tokens.append(token)
            t = torch.from_numpy(np.stack(tokens))
            state = encoder(input_ids=t[:, 0], attention_mask=t[:, 1],
                            token_type_ids=t[:, 2]).last_hidden_state.numpy()
            meta = json.loads((CACHE / f"ru_gap_text_seed{seed}.json").read_text(encoding="utf-8"))
            indices = [meta["lookup"][ru.gap_key(entry)] for entry in entries]
            saved = np.load(CACHE / f"ru_gap_text_seed{seed}.npy", mmap_mode="r")[indices]
            if not np.allclose(state, saved, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"seed {seed} gap BERT cache content mismatch")
    return data, mask_hashes


def entries_for(seed: int):
    rows = [json.loads(line) for line in ru.mask_path(seed).read_text(encoding="utf-8").splitlines()]
    train = {x["row"]: x for x in rows if x["phase"] == "train"}
    phases = {ph: {sc: {x["row"]: x for x in rows
                         if x["phase"] == ph and x["scenario"] == sc}
                   for sc in ru.SCENARIOS} for ph in ("selection", "audit")}
    return train, phases


def run_one(level: str, seed: int, data: dict, mask_hash: str):
    if level not in ("M0", "M1") or seed not in SEEDS:
        raise RuntimeError("invalid frozen run")
    ru.set_determinism(seed)
    run = BASE / f"{level}_seed{seed}"
    if run.exists():
        metrics_path = run / "metrics.json"
        if metrics_path.exists():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result.get("status") == "COMPLETE":
                if digest(run / "mask_manifest.jsonl") != mask_hash:
                    raise RuntimeError("saved mask mismatch")
                print(json.dumps({"run": run.name, "status": "ALREADY_COMPLETE"}), flush=True)
                return result
        raise RuntimeError(f"existing incomplete run needs review: {run}")
    run.mkdir(parents=True)
    shutil.copyfile(ru.mask_path(seed), run / "mask_manifest.jsonl")
    config = {"run_id": run.name, "level": level, "seed": seed, "source": "cleaning_2026e_v2/unaligned train+valid",
              "native_shapes": {"text": [50, 768], "audio": [500, 74], "vision": [500, 35]},
              "model": "text-query cross-attention to audio and vision", "width": DIM, "heads": HEADS,
              "cross_layers_per_direction": 1, "feed_forward": FEED, "dropout": DROP,
              "adapter": level == "M1", "batch": BATCH, "epochs_cap": EPOCHS, "patience": PATIENCE,
              "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "lambda_reg": LAM,
              "selection": "half clean plus half six-gap mean loss", "mask_sha256": mask_hash,
              "plan_sha256": digest(PLAN)}
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
           "ram_total_bytes": psutil.virtual_memory().total,
           "code_sha256": digest(Path(__file__)), "ru_code_sha256": digest(Path(ru.__file__)),
           "v2_manifest_sha256": digest(ru.ROOT / "manifest.json"),
           "scaler_sha256": digest(ru.UNALIGNED / "scalers.json"),
           "bert_weights_sha256": digest(ru.BERT / "pytorch_model.bin"),
           "bert_vocab_sha256": digest(ru.BERT / "vocab.txt"),
           "mask_sha256": mask_hash, "allowed_splits": ["train", "valid"]}
    (run / "environment_checksums.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    for name in ("failures.jsonl", "training_log.jsonl", "predictions_valid.jsonl"):
        (run / name).write_text("", encoding="utf-8")
    train, valid = data["train"], data["valid"]
    clean_train = np.load(CACHE / "clean_train.npy", mmap_mode="r")
    clean_valid = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    gap_meta_path = CACHE / f"ru_gap_text_seed{seed}.json"
    gap_meta = json.loads(gap_meta_path.read_text(encoding="utf-8"))
    if gap_meta["mask_sha256"] != mask_hash or gap_meta["bert_weights_sha256"] != ru.EXPECTED["bert_weights_sha256"]:
        raise RuntimeError("RU gap BERT cache provenance mismatch")
    gap_states = np.load(CACHE / f"ru_gap_text_seed{seed}.npy", mmap_mode="r")
    if gap_states.shape != (gap_meta["unique_sequences"], 50, 768):
        raise RuntimeError("RU gap BERT cache shape mismatch")
    lookup = gap_meta["lookup"]
    train_entries, phase_entries = entries_for(seed)
    specs = [(i, None) for i in range(len(train["samples"]))] + [(i, train_entries[i]) for i in range(len(train["samples"]))]
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    model = NativeMulT(level == "M1")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, best_epoch, bad = float("inf"), 0, 0
    epoch_times = []
    start_wall = time.perf_counter()
    peak = ru.current_rss()
    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.train()
            order = np.random.default_rng(ru.stable_seed("bridge_shuffle", seed, epoch)).permutation(len(specs))
            totals = np.zeros(3)
            for start in range(0, len(order), BATCH):
                chosen = [specs[int(j)] for j in order[start:start + BATCH]]
                batch = ru.stack_batch([ru.build_input(train, clean_train, gap_states, lookup, row, entry)
                                        for row, entry in chosen])
                cy = torch.tensor([int(train["samples"][row]["classification_label"]) for row, _ in chosen])
                ry = torch.tensor([float(train["samples"][row]["regression_label"]) for row, _ in chosen], dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                z, y, _ = model(*batch)
                loss, ce, hub = ru.combined_loss(z, y, cy, ry, LAM)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                opt.step()
                totals += np.array([float(loss), float(ce), float(hub)]) * len(chosen)
                if start % (BATCH * 24) == 0:
                    peak = max(peak, ru.current_rss())
                    if peak > MAX_RSS:
                        raise RuntimeError("Q2_BRIDGE_BUDGET_BLOCKED: RSS")
            sel, preds = ru.evaluate(model, valid, clean_valid, gap_states, lookup,
                                     phase_entries["selection"], yc, yr)
            target_c = torch.from_numpy(yc.astype(np.int64))
            target_r = torch.from_numpy(yr.astype(np.float32))
            losses = [float(ru.combined_loss(torch.from_numpy(z), torch.from_numpy(y), target_c, target_r, LAM)[0])
                      for z, y in preds.values()]
            score = 0.5 * losses[0] + 0.5 * float(np.mean(losses[1:]))
            elapsed = time.perf_counter() - epoch_start
            epoch_times.append(elapsed)
            peak = max(peak, ru.current_rss())
            append(run / "training_log.jsonl", {"epoch": epoch, "train_loss": (totals / len(specs)).tolist(),
                                                 "selection_score": score, "selection_clean": sel["clean"],
                                                 "selection_scenarios": sel["scenarios"],
                                                 "epoch_seconds": elapsed, "rss_bytes": peak})
            print(json.dumps({"run": run.name, "epoch": epoch, "seconds": round(elapsed, 1),
                              "selection_score": round(score, 4), "clean_f1": round(sel["clean"]["macro_f1"], 4),
                              "clean_mae": round(sel["clean"]["mae"], 4)}), flush=True)
            if score < best - 1e-6:
                best, best_epoch, bad = score, epoch, 0
                torch.save(model.state_dict(), run / "checkpoint.pt")
            else:
                bad += 1
            if peak > MAX_RSS or (time.perf_counter() - start_wall + max(epoch_times) * max(0, EPOCHS - epoch)) > MAX_SECONDS:
                raise RuntimeError("Q2_BRIDGE_BUDGET_BLOCKED: projected seed cost")
            if bad >= PATIENCE:
                break
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        audit_start = time.perf_counter()
        audit, preds = ru.evaluate(model, valid, clean_valid, gap_states, lookup,
                                   phase_entries["audit"], yc, yr)
        inference_sec = time.perf_counter() - audit_start
        ru.write_predictions(run, valid, preds)
        if sum(1 for _ in (run / "predictions_valid.jsonl").open(encoding="utf-8")) != 728 * 7:
            raise RuntimeError("incomplete valid prediction file")
        result = {"status": "COMPLETE", "run_id": run.name, "level": level, "seed": seed,
                  "audit": audit, "best_epoch": best_epoch, "selection_score": best,
                  "epoch_times_seconds": epoch_times, "mean_epoch_seconds": float(np.mean(epoch_times)),
                  "train_wall_seconds": time.perf_counter() - start_wall,
                  "audit_inference_seconds": inference_sec, "peak_rss_bytes": peak,
                  "mask_sha256": mask_hash, "checkpoint_sha256": digest(run / "checkpoint.pt")}
        (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": run.name, "status": "COMPLETE", "clean": audit["clean"],
                          "peak_GiB": round(peak / 1024 ** 3, 2)}), flush=True)
        return result
    except Exception as exc:
        failure = {"run_id": run.name, "error": repr(exc),
                   "elapsed_seconds": time.perf_counter() - start_wall,
                   "peak_rss_bytes": peak, "completed_epochs": len(epoch_times)}
        append(run / "failures.jsonl", failure)
        (run / "metrics.json").write_text(json.dumps({"status": "FAILED", **failure}, indent=2), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["M0", "M1"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    data, hashes = preflight()
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "preflight.json").write_text(json.dumps({"status": "PASS", "mask_sha256": hashes,
        "plan_sha256": digest(PLAN), "code_sha256": digest(Path(__file__)),
        "source": "cleaning_2026e_v2/unaligned train+valid"}, indent=2), encoding="utf-8")
    print(json.dumps({"preflight": "PASS", "masks": hashes}), flush=True)
    if args.preflight:
        return
    if args.level is None or args.seed not in SEEDS:
        raise RuntimeError("specify frozen --level and --seed")
    run_one(args.level, args.seed, data, hashes[str(args.seed)])


if __name__ == "__main__":
    main()
