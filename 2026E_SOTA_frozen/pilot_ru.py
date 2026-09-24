"""2026E Stage 4 RU pilot: unaligned_500 native interface, deterministic native masks.

Same sample IDs / seeds / scenario names / support ratios / optimizer & loss policy
as the aligned pilot, but masks are generated on native coordinates:
text 50 tokens, audio 500x74, vision 500x35. Never truncates vision from
vision_lengths. Does not claim byte-identical masks to aligned. Never touches
A2 test or Attachments 3/4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.nn import functional as F

ROOT = Path("workspace/data/processed/cleaning_2026e_v2")
UNALIGNED = ROOT / "unaligned"
BERT = Path("workspace/references/bert-base-uncased")
EXP = Path("workspace/experiments")
MASK_DIR = EXP / "pilot_masks"
RUN_DIR = EXP / "pilot_runs"
RESULT = Path("workspace/results/pilot")
CACHE = EXP / "pilot_cache"
SEEDS = (1729, 2718, 31415)
SHAPES = {"train": 3395, "valid": 728}
VERSION = "unaligned_500"
MODS = ("text", "audio", "vision")
# native time length per modality
LEN = {"text": 50, "audio": 500, "vision": 500}
FEAT = {"text": 768, "audio": 74, "vision": 35}
SCENARIOS = {
    "S1": ("text", "early", 0.10),
    "S2": ("audio", "middle", 0.25),
    "S3": ("vision", "late", 0.40),
    "S4": ("text+audio", "late", 0.25),
    "S5": ("text+vision", "middle", 0.40),
    "S6": ("audio+vision", "early", 0.10),
}
EXPECTED = {
    "contract_sha256": "8cbc834a96275a08ab7670e6c40b1fd6060ebe1551856e9e1ee4e617aebf74c1",
    "scaler_sha256": "6651293905a63bf70f05d2257ca43cb7e5ff44745fb5242485f8ce393470b724",
    "bert_weights_sha256": "0974173802d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a",
    "bert_vocab_sha256": "b49e80874c5efbc7f4cde245a812673b05ef1360ced56b8bc8c750eaeef3fe0f",
}
BATCH_HEAD = 64
BATCH_BERT = 16
MAX_RSS_BYTES = 14 * 1024 ** 3
MAX_SEED_SECONDS = 7200
ACCESS_LOG = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def allowed_read(path: Path) -> Path:
    path = path.resolve()
    try:
        path.relative_to(UNALIGNED.resolve())
        under = True
    except ValueError:
        under = False
    try:
        path.relative_to(BERT.resolve())
        under_bert = True
    except ValueError:
        under_bert = False
    if under:
        if path.parent.name not in ("train", "valid") and path.name != "scalers.json":
            raise RuntimeError(f"forbidden split path: {path}")
    elif under_bert:
        pass
    elif path == (ROOT / "manifest.json").resolve():
        pass
    else:
        raise RuntimeError(f"unexpected input path: {path}")
    ACCESS_LOG.append(str(path))
    return path


def load_split(split: str) -> dict:
    if split not in SHAPES:
        raise RuntimeError(f"forbidden split: {split}")
    base = UNALIGNED / split
    out = {}
    for name in ("text_bert", "audio", "vision",
                 "text_P", "text_O", "audio_P", "audio_O", "vision_P", "vision_O"):
        out[name] = np.load(allowed_read(base / f"{name}.npy"), allow_pickle=False)
    out["samples"] = json.loads(allowed_read(base / "samples.json").read_text(encoding="utf-8"))
    return out


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def eligible_positions(data: dict, row: int, mod: str) -> np.ndarray:
    eligible = (data[f"{mod}_P"][row] == 1) & (data[f"{mod}_O"][row] == 1)
    if mod == "text":
        ids = data["text_bert"][row, 0]
        eligible &= (ids != 0) & (ids != 101) & (ids != 102)
    return np.flatnonzero(eligible)


def band_indices(positions: np.ndarray, band: str) -> np.ndarray:
    thirds = np.array_split(positions, 3)
    return thirds[{"early": 0, "middle": 1, "late": 2}[band]]


def possible_spans(indices: np.ndarray, length: int):
    if length < 1:
        return []
    values = set(map(int, indices))
    return [(s, s + length) for s in sorted(values)
            if all(t in values for t in range(s, s + length))]


def select_span(indices: np.ndarray, length: int, rng):
    if len(indices) == 0:
        return None
    for n in range(min(length, len(indices)), 0, -1):
        spans = possible_spans(indices, n)
        if spans:
            return spans[int(rng.integers(len(spans)))]
    return None


def scenario_mods(label: str):
    return tuple(label.split("+"))


def make_mask(data: dict, split: str, row: int, scenario: str, seed: int, phase: str) -> dict:
    label, band, ratio = SCENARIOS[scenario]
    mods = scenario_mods(label)
    sample_id = data["samples"][row]["id"]
    rng = np.random.default_rng(stable_seed(VERSION, split, sample_id, scenario, seed, phase))
    positions = {m: eligible_positions(data, row, m) for m in mods}
    bands = {m: band_indices(positions[m], band) for m in mods}
    wanted = {m: max(1, int(round(ratio * len(positions[m])))) if len(positions[m]) else 0
              for m in mods}
    spans = {}
    # joint common-index only when both modalities share native length
    if len(mods) == 2 and LEN[mods[0]] == LEN[mods[1]]:
        common = np.intersect1d(bands[mods[0]], bands[mods[1]])
        joint = select_span(common, min(wanted.values()), rng) if all(wanted.values()) else None
        if joint is not None:
            spans = {m: joint for m in mods}
    if not spans:
        spans = {m: select_span(bands[m], wanted[m], rng) for m in mods}
    injectable = all(spans[m] is not None for m in mods)
    if not injectable:
        spans = {m: None for m in mods}
    overlap = None
    if len(mods) == 2 and injectable and LEN[mods[0]] == LEN[mods[1]]:
        overlap = max(0, min(spans[mods[0]][1], spans[mods[1]][1]) -
                      max(spans[mods[0]][0], spans[mods[1]][0]))
    return {
        "version": VERSION, "split": split, "phase": phase, "row": row,
        "sample_id": sample_id, "scenario": scenario, "seed": seed,
        "modalities": mods, "position_band": band, "requested_ratio": ratio,
        "native_lengths": {m: LEN[m] for m in mods},
        "spans": {m: list(spans[m]) if spans[m] is not None else None for m in mods},
        "eligible_counts": {m: len(positions[m]) for m in mods},
        "realized_ratios": {m: (spans[m][1] - spans[m][0]) / len(positions[m])
                            if spans[m] is not None and len(positions[m]) else 0.0
                            for m in mods},
        "overlap_native": overlap,
        "cross_time_alignment_asserted": bool(len(mods) == 2 and LEN[mods[0]] != LEN[mods[1]]) is False,
        "injectable": injectable,
    }


def mask_rows(train: dict, valid: dict, seed: int):
    rows = []
    for i, sample in enumerate(train["samples"]):
        sc = list(SCENARIOS)[stable_seed("train_assignment", sample["id"], seed) % 6]
        rows.append(make_mask(train, "train", i, sc, seed, "train"))
    for phase in ("selection", "audit"):
        for scenario in SCENARIOS:
            for i in range(len(valid["samples"])):
                rows.append(make_mask(valid, "valid", i, scenario, seed, phase))
    return rows


def serialize_masks(rows) -> bytes:
    return b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
                    for r in rows)


def mask_path(seed: int) -> Path:
    return MASK_DIR / f"unaligned_seed{seed}.jsonl"


def build_masks(train, valid, seed: int) -> dict:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = mask_rows(train, valid, seed)
    payload = serialize_masks(rows)
    assert payload == serialize_masks(mask_rows(train, valid, seed))
    path = mask_path(seed)
    if path.exists():
        assert path.read_bytes() == payload, "existing unaligned mask mismatch"
    else:
        path.write_bytes(payload)
    stats = {}
    for phase in ("train", "selection", "audit"):
        sub = [r for r in rows if r["phase"] == phase]
        stats[phase] = {"rows": len(sub), "injectable": sum(r["injectable"] for r in sub)}
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "stats": stats}


def apply_mask(data: dict, row: int, entry: dict) -> dict:
    out = {name: data[name][row].copy() for name in
           ("text_bert", "audio", "vision",
            "text_P", "text_O", "audio_P", "audio_O", "vision_P", "vision_O")}
    injected = {m: np.zeros(LEN[m], dtype=bool) for m in MODS}
    if entry["injectable"]:
        for mod, span in entry["spans"].items():
            a, b = span
            idx = eligible_positions(data, row, mod)
            assert all(t in idx for t in range(a, b)), "mask touched untrusted position"
            if mod == "text":
                out["text_bert"][0, a:b] = 100
            else:
                out[mod][a:b] = 0.0
            out[f"{mod}_O"][a:b] = 0
            injected[mod][a:b] = True
    out["injected_mask"] = injected
    return out


def current_rss() -> int:
    proc = psutil.Process()
    return proc.memory_info().rss + sum(c.memory_info().rss for c in
                                        proc.children(recursive=True) if c.is_running())


def set_determinism(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)


def cache_gap_bert_ru(train, valid, seed: int) -> dict:
    """Encode RU text-intervened token sequences (native masks) after [UNK]."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"ru_gap_text_seed{seed}.npy"
    meta_path = CACHE / f"ru_gap_text_seed{seed}.json"
    mask_sha = sha256(mask_path(seed))
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["mask_sha256"] == mask_sha
        assert np.load(path, mmap_mode="r").shape == (meta["unique_sequences"], 50, 768)
        return {**meta, "reused": True, "path": str(path)}
    rows = [json.loads(l) for l in mask_path(seed).read_text(encoding="utf-8").splitlines()]
    unique, tokens, lookup = {}, [], {}
    for entry in rows:
        if not entry["injectable"] or "text" not in entry["modalities"]:
            continue
        data = train if entry["split"] == "train" else valid
        token = data["text_bert"][entry["row"]].copy()
        a, b = entry["spans"]["text"]
        token[0, a:b] = 100
        key = token.tobytes()
        if key not in unique:
            unique[key] = len(tokens)
            tokens.append(token)
        lookup[gap_key(entry)] = unique[key]
    from transformers import BertModel
    set_determinism(seed)
    t0 = time.perf_counter()
    model = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(tokens), 50, 768))
    peak = current_rss()
    with torch.inference_mode():
        for start in range(0, len(tokens), BATCH_BERT):
            t = torch.from_numpy(np.stack(tokens[start:start + BATCH_BERT]))
            h = model(input_ids=t[:, 0], attention_mask=t[:, 1],
                      token_type_ids=t[:, 2]).last_hidden_state
            out[start:start + len(t)] = h.numpy()
            peak = max(peak, current_rss())
            if peak > MAX_RSS_BYTES:
                raise RuntimeError("RU gap BERT cache RAM >14 GiB")
        out.flush()
    elapsed = time.perf_counter() - t0
    meta = {"seed": seed, "mask_sha256": mask_sha,
            "bert_weights_sha256": EXPECTED["bert_weights_sha256"],
            "unique_sequences": len(tokens), "lookup": lookup,
            "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
            "reused": False, "path": str(path)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def gap_key(entry: dict) -> str:
    return f"{entry['split']}|{entry['phase']}|{entry['scenario']}|{entry['row']}"


class RUHead(nn.Module):
    """Minimal B1-style modality-native pooling baseline; no alignment network."""

    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(FEAT[m], 128) for m in MODS])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(LEN[m], 128) * 0.02)
                                     for m in MODS])
        self.p_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in MODS])
        self.o_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in MODS])
        self.fuse = nn.Sequential(nn.Linear(384, 128), nn.GELU())
        self.cls = nn.Linear(128, 3)
        self.reg = nn.Linear(128, 1)

    @staticmethod
    def pool(x, mask):
        w = mask.float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(self, text_state, audio, vision, p_list, o_list, unknown_excluded=False):
        xs = (text_state, audio, vision)
        hs = []
        for mi, m in enumerate(MODS):
            p, o = p_list[mi], o_list[mi]
            base = self.proj[mi](xs[mi])
            pos = self.pos[mi].unsqueeze(0)
            state = self.p_embed[mi](p.long()) + self.o_embed[mi](o.long()) + pos
            z = F.gelu(base + state)
            trusted = (p == 1) & (o == 1)
            content = self.pool(z, trusted)
            # state summary over every non-padding position (retains native unknown=2)
            state_summary = self.pool(F.gelu(state), p != 0)
            hs.append(content + state_summary)
        h = self.fuse(torch.cat(hs, dim=1))
        return self.cls(h), 3.0 * torch.tanh(self.reg(h)).squeeze(1), None


def build_input(data: dict, clean_states, gap_states, lookup, row: int, entry):
    if entry is None:
        t = np.asarray(clean_states[row])
        a, v = data["audio"][row], data["vision"][row]
        p = [data[f"{m}_P"][row] for m in MODS]
        o = [data[f"{m}_O"][row] for m in MODS]
        return t, a, v, p, o
    item = apply_mask(data, row, entry)
    key = gap_key(entry)
    t = np.asarray(gap_states[lookup[key]] if key in lookup else clean_states[row])
    a, v = item["audio"], item["vision"]
    p = [item[f"{m}_P"] for m in MODS]
    o = [item[f"{m}_O"] for m in MODS]
    return t, a, v, p, o


def stack_batch(parts):
    ts, as_, vs, ps, os_ = [], [], [], {m: [] for m in MODS}, {m: [] for m in MODS}
    for (t, a, v, p, o) in parts:
        ts.append(t); as_.append(a); vs.append(v)
        for mi, m in enumerate(MODS):
            ps[m].append(p[mi]); os_[m].append(o[mi])
    p_list = [torch.from_numpy(np.stack(ps[m]).copy()) for m in MODS]
    o_list = [torch.from_numpy(np.stack(os_[m]).copy()) for m in MODS]
    return (torch.from_numpy(np.stack(ts).copy()),
            torch.from_numpy(np.stack(as_).copy()),
            torch.from_numpy(np.stack(vs).copy()), p_list, o_list)


def forward_batch(model, data, clean_states, gap_states, lookup, specs, unknown_excluded=False):
    parts = [build_input(data, clean_states, gap_states, lookup, r, e) for (r, e) in specs]
    t, a, v, p, o = stack_batch(parts)
    return model(t, a, v, p, o, unknown_excluded=unknown_excluded)


def targets_for(rows):
    # targets are read from a fixed samples list supplied by caller closure
    raise NotImplementedError


def metrics_from_arrays(yc, yr, logits, pred_reg):
    pred_cls = logits.argmax(axis=1)
    pearson = None
    if len(yr) >= 2 and np.std(yr) != 0 and np.std(pred_reg) != 0:
        pearson = float(np.corrcoef(yr, pred_reg)[0, 1])
    return {"n": len(yc), "accuracy": float(accuracy_score(yc, pred_cls)),
            "macro_f1": float(f1_score(yc, pred_cls, average="macro",
                                      labels=[0, 1, 2], zero_division=0)),
            "weighted_f1": float(f1_score(yc, pred_cls, average="weighted",
                                         labels=[0, 1, 2], zero_division=0)),
            "mae": float(mean_absolute_error(yr, pred_reg)), "pearson": pearson,
            "predicted_class_counts": {str(k): int((pred_cls == k).sum()) for k in range(3)}}


def predict_all(model, data, clean_states, gap_states, lookup, entries,
                unknown_excluded=False):
    model.eval()
    logits, intensity = [], []
    n = len(data["samples"])
    with torch.inference_mode():
        for start in range(0, n, BATCH_HEAD):
            rows = list(range(start, min(n, start + BATCH_HEAD)))
            specs = [(r, entries[r] if entries is not None else None) for r in rows]
            z, y, _ = forward_batch(model, data, clean_states, gap_states,
                                    lookup, specs, unknown_excluded)
            logits.append(z.numpy()); intensity.append(y.numpy())
    return np.concatenate(logits), np.concatenate(intensity)


def scenario_metrics(clean_z, clean_y, gap_z, gap_y, entries, yc, yr):
    eligible = np.array([entries[i]["injectable"] for i in range(len(entries))], dtype=bool)
    if not eligible.any():
        return {"n_eligible": 0, "n_not_injectable": int(len(eligible)),
                "status": "NO_ELIGIBLE_SAMPLES"}
    clean = metrics_from_arrays(yc[eligible], yr[eligible],
                                clean_z[eligible], clean_y[eligible])
    gap = metrics_from_arrays(yc[eligible], yr[eligible],
                              gap_z[eligible], gap_y[eligible])
    delta = {k: (gap[k] - clean[k]) if gap[k] is not None and clean[k] is not None else None
             for k in ("accuracy", "macro_f1", "mae", "pearson")}
    return {"n_eligible": int(eligible.sum()),
            "n_not_injectable": int((~eligible).sum()),
            "clean_paired": clean, "gap_paired": gap, "delta": delta}


def evaluate(model, data, clean_states, gap_states, lookup, phase_entries,
             yc, yr, unknown_excluded=False):
    clean_z, clean_y = predict_all(model, data, clean_states, gap_states,
                                   lookup, None, unknown_excluded)
    out = {"clean": metrics_from_arrays(yc, yr, clean_z, clean_y), "scenarios": {}}
    preds = {"clean": (clean_z, clean_y)}
    for scenario, entries in phase_entries.items():
        gz, gy = predict_all(model, data, clean_states, gap_states,
                             lookup, entries, unknown_excluded)
        out["scenarios"][scenario] = scenario_metrics(
            clean_z, clean_y, gz, gy, entries, yc, yr)
        preds[scenario] = (gz, gy)
    return out, preds


def combined_loss(logits, intensity, cy, ry, lam):
    ce = F.cross_entropy(logits, cy)
    huber = F.huber_loss(intensity, ry, delta=1.0)
    return ce + lam * huber, ce, huber


def write_predictions(run: Path, data, preds):
    with (run / "predictions_valid.jsonl").open("w", encoding="utf-8") as f:
        for cond, (z, y) in preds.items():
            for i, s in enumerate(data["samples"]):
                rec = {"split": "valid", "condition": cond, "row": i,
                       "sample_id": s["id"],
                       "true_class": int(s["classification_label"]),
                       "true_intensity": float(s["regression_label"]),
                       "logits": [float(v) for v in z[i]],
                       "predicted_class": int(z[i].argmax()),
                       "predicted_intensity": float(y[i])}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path, rec):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def init_run(seed, lam):
    run_id = f"RU_seed{seed}"
    run = RUN_DIR / run_id
    if run.exists():
        return run, run_id, True
    run.mkdir(parents=True)
    shutil.copyfile(mask_path(seed), run / "mask_manifest.jsonl")
    config = {"run_id": run_id, "model_level": "RU", "seed": seed,
              "lambda_cls": 1.0, "lambda_reg": lam, "epochs_cap": 12,
              "batch_size": BATCH_HEAD, "optimizer": "AdamW",
              "learning_rate": 0.001, "weight_decay": 0.0001, "huber_delta": 1.0,
              "condition": "clean_plus_one_gap_copy_native",
              "source_version": "cleaning_2026e_v2/unaligned",
              "interface": VERSION, "bert_frozen": True,
              "unknown_policy": "retained_native", "alignment_network": False,
              "vision_truncated_from_lengths": False,
              "mask_sha256": sha256(run / "mask_manifest.jsonl")}
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "cpu_threads": torch.get_num_threads(),
           "ram_total_bytes": psutil.virtual_memory().total,
           "code_sha256": sha256(Path(__file__)),
           "v2_manifest_sha256": sha256(ROOT / "manifest.json"),
           "scaler_sha256": EXPECTED["scaler_sha256"],
           "bert_weights_sha256": EXPECTED["bert_weights_sha256"],
           "bert_vocab_sha256": EXPECTED["bert_vocab_sha256"],
           "mask_sha256": config["mask_sha256"],
           "allowed_input_splits": ["train", "valid"]}
    (run / "environment_checksums.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8")
    (run / "failures.jsonl").write_text("", encoding="utf-8")
    (run / "training_log.jsonl").write_text("", encoding="utf-8")
    (run / "predictions_valid.jsonl").write_text("", encoding="utf-8")
    return run, run_id, False


def native_gap_summary(rows) -> dict:
    """Aggregate not_injectable counts and realized ratios from native masks."""
    per_scenario = {}
    for sc in SCENARIOS:
        sub = [r for r in rows if r["phase"] == "audit" and r["scenario"] == sc]
        per_scenario[sc] = {
            "n_not_injectable": sum(not r["injectable"] for r in sub),
            "mean_realized_ratio": float(np.mean([
                np.mean(list(r["realized_ratios"].values())) for r in sub
                if r["injectable"]])) if any(r["injectable"] for r in sub) else 0.0}
    return per_scenario


def fit_ru(seed: int):
    set_determinism(seed)
    train, valid = load_split("train"), load_split("valid")
    build_masks(train, valid, seed)
    lam = json.loads((RESULT / "lambda_selection.json").read_text(
        encoding="utf-8"))["lambda_reg"]
    gap_meta = cache_gap_bert_ru(train, valid, seed)
    lookup = gap_meta["lookup"]
    # clean BERT states are identical to aligned (text_bert identical) -> reuse
    clean_train = np.load(CACHE / "clean_train.npy", mmap_mode="r")
    clean_valid = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(CACHE / f"ru_gap_text_seed{seed}.npy", mmap_mode="r")
    run, run_id, existed = init_run(seed, lam)
    if existed and (run / "metrics.json").exists():
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if m.get("status") == "COMPLETE":
            print(json.dumps({"run": run_id, "status": "ALREADY_COMPLETE"}))
            return m
    t0 = time.perf_counter()
    peak = current_rss()
    model = RUHead()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n = len(train["samples"])
    specs = [(i, None) for i in range(n)] + [
        (i, r) for i, r in enumerate([
            x for x in (json.loads(l) for l in
                        mask_path(seed).read_text(encoding="utf-8").splitlines())
            if x["phase"] == "train"])]
    train_entries = {x["row"]: x for x in (
        json.loads(l) for l in mask_path(seed).read_text(encoding="utf-8").splitlines())
        if x["phase"] == "train"}
    specs = [(i, None) for i in range(n)] + [
        (i, train_entries[i]) for i in range(n)]
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    phase_entries = {ph: {sc: {x["row"]: x for x in (
        json.loads(l) for l in mask_path(seed).read_text(encoding="utf-8").splitlines())
        if x["phase"] == ph and x["scenario"] == sc} for sc in SCENARIOS}
        for ph in ("selection", "audit")}
    best, best_epoch, no_improve = float("inf"), 0, 0
    epoch_times = []
    try:
        for epoch in range(1, 13):
            et = time.perf_counter()
            model.train()
            order = np.random.default_rng(stable_seed(
                "shuffle", run_id, seed, epoch)).permutation(len(specs))
            accum = np.zeros(3)
            for start in range(0, len(order), BATCH_HEAD):
                chosen_idx = order[start:start + BATCH_HEAD]
                chosen = [specs[int(i)] for i in chosen_idx]
                t, a, v, p, o = stack_batch([
                    build_input(train, clean_train, gap_states, lookup, r, e)
                    for (r, e) in chosen])
                rows = np.array([r for r, _ in chosen])
                cy = torch.tensor([int(train["samples"][i]["classification_label"])
                                   for i in rows], dtype=torch.long)
                ry = torch.tensor([float(train["samples"][i]["regression_label"])
                                   for i in rows], dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                z, y, _ = model(t, a, v, p, o)
                loss, ce, hub = combined_loss(z, y, cy, ry, lam)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite train loss")
                loss.backward(); opt.step()
                accum += np.array([float(loss), float(ce), float(hub)]) * len(rows)
            sel_out, sel_pred = evaluate(model, valid, clean_valid, gap_states,
                                         lookup, phase_entries["selection"], yc, yr)
            # composite selection score: 0.5 clean + 0.5 mean scenarios
            comp_rows = np.arange(len(valid["samples"]))
            ccy = torch.tensor(yc, dtype=torch.long)
            cry = torch.tensor(yr, dtype=torch.float32)
            cond_vals = []
            for cond, (z, y) in sel_pred.items():
                l, _, _ = combined_loss(torch.from_numpy(z), torch.from_numpy(y),
                                        ccy, cry, lam)
                cond_vals.append(float(l))
            score = 0.5 * cond_vals[0] + 0.5 * float(np.mean(cond_vals[1:]))
            epoch_sec = time.perf_counter() - et
            epoch_times.append(epoch_sec)
            peak = max(peak, current_rss())
            append_jsonl(run / "training_log.jsonl",
                         {"epoch": epoch, "train_loss": (accum / len(specs)).tolist(),
                          "selection_score": score,
                          "selection_clean": sel_out["clean"],
                          "selection_scenarios": sel_out["scenarios"],
                          "epoch_seconds": epoch_sec, "rss_bytes": peak})
            print(json.dumps({"run": run_id, "epoch": epoch,
                              "seconds": round(epoch_sec, 1),
                              "clean_f1": round(sel_out["clean"]["macro_f1"], 4),
                              "clean_mae": round(sel_out["clean"]["mae"], 4)},
                             ensure_ascii=False), flush=True)
            if score < best - 1e-6:
                best, best_epoch, no_improve = score, epoch, 0
                torch.save(model.state_dict(), run / "checkpoint.pt")
            else:
                no_improve += 1
            if peak > MAX_RSS_BYTES or (time.perf_counter() - t0 +
                                        max(epoch_times) * max(0, 12 - epoch)) > MAX_SEED_SECONDS:
                raise RuntimeError("PILOT_BUDGET_BLOCKED: RAM/time cap")
            if no_improve >= 3:
                break
        model.load_state_dict(torch.load(run / "checkpoint.pt",
                                         map_location="cpu", weights_only=True))
        inf_t0 = time.perf_counter()
        audit, preds = evaluate(model, valid, clean_valid, gap_states,
                                lookup, phase_entries["audit"], yc, yr)
        inf_wall = time.perf_counter() - inf_t0
        write_predictions(run, valid, preds)
        mask_audit_rows = [json.loads(l) for l in
                           mask_path(seed).read_text(encoding="utf-8").splitlines()]
        result = {"status": "COMPLETE", "run_id": run_id, "level": "RU",
                  "seed": seed, "lambda_reg": lam, "audit": audit,
                  "best_epoch": best_epoch, "best_selection_score": best,
                  "epoch_times_seconds": epoch_times,
                  "mean_epoch_seconds": float(np.mean(epoch_times)),
                  "train_wall_seconds": time.perf_counter() - t0,
                  "audit_inference_seconds": inf_wall,
                  "ru_gap_bert_cache_seconds": gap_meta["elapsed_seconds"],
                  "ru_gap_bert_cache_reused": gap_meta["reused"],
                  "clean_bert_cache_reused": True,
                  "peak_rss_bytes": peak,
                  "native_gap_summary": native_gap_summary(mask_audit_rows),
                  "checkpoint_sha256": sha256(run / "checkpoint.pt")}
        (run / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": run_id, "COMPLETE": True,
                          "clean": audit["clean"],
                          "mean_epoch": round(result["mean_epoch_seconds"], 1),
                          "peak_GB": round(peak / 1024 ** 3, 2)}, ensure_ascii=False))
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "error": repr(exc),
                   "elapsed_seconds": time.perf_counter() - t0,
                   "peak_rss_bytes": peak,
                   "completed_epochs": len(epoch_times)}
        append_jsonl(run / "failures.jsonl", failure)
        (run / "metrics.json").write_text(
            json.dumps({"status": "FAILED", **failure}, indent=2), encoding="utf-8")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["ru"])
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit("seed must be a frozen seed")
    fit_ru(args.seed)


if __name__ == "__main__":
    main()
