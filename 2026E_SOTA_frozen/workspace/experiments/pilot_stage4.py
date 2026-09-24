"""2026E Stage 4 pilot: frozen aligned-v2 input, deterministic gaps, sequential runs.

Never accesses Attachment 2 test or Attachments 3/4. No raw/processed writes.
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
ALIGNED = ROOT / "aligned"
BERT = Path("workspace/references/bert-base-uncased")
EXP = Path("workspace/experiments")
MASK_DIR = EXP / "pilot_masks"
RUN_DIR = EXP / "pilot_runs"
RESULT = Path("workspace/results/pilot")
SEEDS = (1729, 2718, 31415)
SHAPES = {"train": 3395, "valid": 728}
SCENARIOS = {
    "S1": ("T", "early", 0.10),
    "S2": ("A", "middle", 0.25),
    "S3": ("V", "late", 0.40),
    "S4": ("TA", "late", 0.25),
    "S5": ("TV", "middle", 0.40),
    "S6": ("AV", "early", 0.10),
}
EXPECTED = {
    "contract_sha256": "8cbc834a96275a08ab7670e6c40b1fd6060ebe1551856e9e1ee4e617aebf74c1",
    "scaler_sha256": "6651293905a63bf70f05d2257ca43cb7e5ff44745fb5242485f8ce393470b724",
    "bert_weights_sha256": "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a",
    "bert_vocab_sha256": "b49e80874c5efbc7f4cde245a812673b05ef1360ced56b8bc8c750eaeef3fe0f",
}
ACCESS_LOG = []
MODS = ("text", "audio", "vision")
CACHE = EXP / "pilot_cache"
BATCH_HEAD = 64
BATCH_BERT = 16
MAX_RSS_BYTES = 14 * 1024 ** 3
MAX_SEED_SECONDS = 7200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def allowed_read(path: Path) -> Path:
    path = path.resolve()
    try:
        path.relative_to(ALIGNED.resolve())
        under_aligned = True
    except ValueError:
        under_aligned = False
    try:
        path.relative_to(BERT.resolve())
        under_bert = True
    except ValueError:
        under_bert = False
    if under_aligned:
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
    base = ALIGNED / split
    out = {}
    for name in ("text_bert", "audio", "vision", "text_P", "text_O", "audio_P", "audio_O", "vision_P", "vision_O"):
        out[name] = np.load(allowed_read(base / f"{name}.npy"), allow_pickle=False)
    out["samples"] = json.loads(allowed_read(base / "samples.json").read_text(encoding="utf-8"))
    return out


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def eligible_positions(data: dict, row: int, mod: str) -> np.ndarray:
    name = {"T": "text", "A": "audio", "V": "vision"}[mod]
    eligible = (data[f"{name}_P"][row] == 1) & (data[f"{name}_O"][row] == 1)
    if mod == "T":
        ids = data["text_bert"][row, 0]
        eligible &= (ids != 0) & (ids != 101) & (ids != 102)
    return np.flatnonzero(eligible)


def band_indices(positions: np.ndarray, band: str) -> np.ndarray:
    thirds = np.array_split(positions, 3)
    return thirds[{"early": 0, "middle": 1, "late": 2}[band]]


def possible_spans(indices: np.ndarray, length: int) -> list[tuple[int, int]]:
    if length < 1:
        return []
    values = set(map(int, indices))
    return [(start, start + length) for start in sorted(values)
            if all(t in values for t in range(start, start + length))]


def select_span(indices: np.ndarray, length: int, rng: np.random.Generator):
    if len(indices) == 0:
        return None
    for n in range(min(length, len(indices)), 0, -1):
        spans = possible_spans(indices, n)
        if spans:
            return spans[int(rng.integers(len(spans)))]
    return None


def make_mask(data: dict, split: str, row: int, scenario: str, seed: int, phase: str) -> dict:
    mods, band, ratio = SCENARIOS[scenario]
    sample_id = data["samples"][row]["id"]
    rng = np.random.default_rng(stable_seed("aligned_50", split, sample_id, scenario, seed, phase))
    positions = {m: eligible_positions(data, row, m) for m in mods}
    bands = {m: band_indices(positions[m], band) for m in mods}
    wanted = {m: max(1, int(round(ratio * len(positions[m])))) if len(positions[m]) else 0 for m in mods}
    spans = {}
    if len(mods) == 2:
        common = np.intersect1d(bands[mods[0]], bands[mods[1]])
        joint = select_span(common, min(wanted.values()), rng) if all(wanted.values()) else None
        if joint is not None:
            spans = {m: joint for m in mods}
    if not spans:
        spans = {m: select_span(bands[m], wanted[m], rng) for m in mods}
    injectable = all(spans[m] is not None for m in mods)
    if not injectable:
        spans = {m: None for m in mods}
    return {
        "split": split, "phase": phase, "row": row, "sample_id": sample_id,
        "scenario": scenario, "seed": seed, "modalities": mods,
        "position_band": band, "requested_ratio": ratio,
        "spans": {m: list(spans[m]) if spans[m] is not None else None for m in mods},
        "eligible_counts": {m: len(positions[m]) for m in mods},
        "realized_ratios": {m: (spans[m][1] - spans[m][0]) / len(positions[m])
                            if spans[m] is not None and len(positions[m]) else 0.0 for m in mods},
        "overlap": (max(0, min(spans[mods[0]][1], spans[mods[1]][1]) -
                         max(spans[mods[0]][0], spans[mods[1]][0]))
                    if len(mods) == 2 and injectable else None),
        "injectable": injectable,
    }


def mask_rows(train: dict, valid: dict, seed: int) -> list[dict]:
    rows = []
    for i, sample in enumerate(train["samples"]):
        scenario = list(SCENARIOS)[stable_seed("train_assignment", sample["id"], seed) % 6]
        rows.append(make_mask(train, "train", i, scenario, seed, "train"))
    for phase in ("selection", "audit"):
        for scenario in SCENARIOS:
            for i in range(len(valid["samples"])):
                rows.append(make_mask(valid, "valid", i, scenario, seed, phase))
    return rows


def serialize_masks(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for r in rows)


def mask_path(seed: int) -> Path:
    return MASK_DIR / f"aligned_seed{seed}.jsonl"


def build_masks(train: dict, valid: dict, seed: int) -> dict:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = mask_rows(train, valid, seed)
    payload = serialize_masks(rows)
    # Determinism check uses a fresh generation, not the same object.
    assert payload == serialize_masks(mask_rows(train, valid, seed))
    path = mask_path(seed)
    if path.exists():
        assert path.read_bytes() == payload, "existing mask manifest mismatch"
    else:
        path.write_bytes(payload)
    stats = {}
    for phase in ("train", "selection", "audit"):
        subset = [r for r in rows if r["phase"] == phase]
        stats[phase] = {"rows": len(subset), "injectable": sum(r["injectable"] for r in subset)}
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "stats": stats}


def apply_mask(data: dict, row: int, entry: dict) -> dict:
    out = {name: data[name][row].copy() for name in
           ("text_bert", "audio", "vision", "text_P", "text_O", "audio_P", "audio_O", "vision_P", "vision_O")}
    injected = {m: np.zeros(50, dtype=bool) for m in "TAV"}
    if entry["injectable"]:
        for mod, span in entry["spans"].items():
            a, b = span
            idx = eligible_positions(data, row, mod)
            assert all(t in idx for t in range(a, b)), "mask touched untrusted position"
            name = {"T": "text", "A": "audio", "V": "vision"}[mod]
            if mod == "T":
                out["text_bert"][0, a:b] = 100
            else:
                out[name][a:b] = 0.0
            out[f"{name}_O"][a:b] = 0
            injected[mod][a:b] = True
    out["injected_mask"] = injected
    return out


def smoke() -> dict:
    start = time.perf_counter()
    RESULT.mkdir(parents=True, exist_ok=True)
    record = {"status": "PENDING", "checks": {}, "accessed_files": []}
    try:
        manifest = json.loads(allowed_read(ROOT / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["contract_sha256"] == EXPECTED["contract_sha256"]
        scaler_path = allowed_read(ALIGNED / "scalers.json")
        assert sha256(scaler_path) == EXPECTED["scaler_sha256"]
        assert sha256(allowed_read(BERT / "pytorch_model.bin")) == EXPECTED["bert_weights_sha256"]
        assert sha256(allowed_read(BERT / "vocab.txt")) == EXPECTED["bert_vocab_sha256"]
        record["checks"]["hashes"] = {"contract": manifest["contract_sha256"],
                                       "scaler": sha256(scaler_path),
                                       "bert_weights": EXPECTED["bert_weights_sha256"],
                                       "bert_vocab": EXPECTED["bert_vocab_sha256"]}
        train, valid = load_split("train"), load_split("valid")
        for split, data in (("train", train), ("valid", valid)):
            n = SHAPES[split]
            assert len(data["samples"]) == n
            assert data["text_bert"].shape == (n, 3, 50)
            assert data["audio"].shape == (n, 50, 74)
            assert data["vision"].shape == (n, 50, 35)
            assert data["text_bert"].dtype == np.int64
            assert np.array_equal(data["text_bert"][:, 1, :], data["text_P"])
            assert np.isfinite(data["audio"]).all() and np.isfinite(data["vision"]).all()
            for mod in ("text", "audio", "vision"):
                p, o = data[f"{mod}_P"], data[f"{mod}_O"]
                assert p.shape == (n, 50) and o.shape == (n, 50)
                assert set(np.unique(p)).issubset({0, 1, 2})
                assert set(np.unique(o)).issubset({0, 1, 2})
                assert np.array_equal(p == 0, o == 0)
            for sample in data["samples"]:
                y = float(sample["regression_label"])
                cls = 0 if y < 0 else 1 if y == 0 else 2
                assert int(sample["classification_label"]) == cls
            record["checks"][split] = {"n": n, "shapes": {k: list(data[k].shape) for k in ("text_bert", "audio", "vision")}}
        from transformers import BertModel, BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(str(BERT), local_files_only=True)
        encoder = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
        with torch.no_grad():
            token = torch.from_numpy(train["text_bert"][:1].copy())
            state = encoder(input_ids=token[:, 0], attention_mask=token[:, 1],
                            token_type_ids=token[:, 2]).last_hidden_state
        assert len(tokenizer) == 30522 and state.shape == (1, 50, 768) and torch.isfinite(state).all()
        record["checks"]["bert_forward"] = {"shape": list(state.shape), "finite": True}
        mask_record = build_masks(train, valid, SEEDS[0])
        rows = [json.loads(x) for x in mask_path(SEEDS[0]).read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3395 + 2 * 6 * 728
        for item in rows:
            data = train if item["split"] == "train" else valid
            out = apply_mask(data, item["row"], item)
            assert sum(x.sum() for x in out["injected_mask"].values()) == sum(
                b - a for v in item["spans"].values() if v is not None for a, b in [v])
        record["checks"]["masks"] = mask_record
        record["checks"]["deterministic_seeds"] = list(SEEDS)
        record["status"] = "PASSED"
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = repr(exc)
    record["accessed_files"] = sorted(set(ACCESS_LOG))
    record["elapsed_seconds"] = time.perf_counter() - start
    record["environment"] = {"python": sys.version, "platform": platform.platform(),
                              "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                              "ram_bytes": psutil.virtual_memory().total,
                              "cpu_count": os.cpu_count()}
    out = RESULT / "smoke_check.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"],
                      "mask_sha256": record.get("checks", {}).get("masks", {}).get("sha256"),
                      "error": record.get("error")}, ensure_ascii=False))
    if record["status"] != "PASSED":
        raise SystemExit(2)
    return record


def current_rss() -> int:
    proc = psutil.Process()
    return proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True) if c.is_running())


def set_determinism(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)


def cache_clean_bert(train: dict, valid: dict) -> dict:
    """Encode exact clean token rows once, without labels or optimizer."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "clean_text_cache.json"
    paths = {split: CACHE / f"clean_{split}.npy" for split in SHAPES}
    tokens_sha = {split: hashlib.sha256(data["text_bert"].tobytes()).hexdigest()
                  for split, data in (("train", train), ("valid", valid))}
    if meta_path.exists() and all(p.exists() for p in paths.values()):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["tokens_sha256"] == tokens_sha
        assert meta["bert_weights_sha256"] == EXPECTED["bert_weights_sha256"]
        for split, path in paths.items():
            assert np.load(path, mmap_mode="r").shape == (SHAPES[split], 50, 768)
        return {**meta, "reused": True, "paths": {k: str(v) for k, v in paths.items()}}
    from transformers import BertModel
    t0 = time.perf_counter()
    model = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
    peak = current_rss()
    with torch.inference_mode():
        for split, data in (("train", train), ("valid", valid)):
            tokens = data["text_bert"]
            out = np.lib.format.open_memmap(paths[split], mode="w+", dtype=np.float32,
                                            shape=(len(tokens), 50, 768))
            for start in range(0, len(tokens), BATCH_BERT):
                t = torch.from_numpy(tokens[start:start+BATCH_BERT].copy())
                h = model(input_ids=t[:, 0], attention_mask=t[:, 1],
                          token_type_ids=t[:, 2]).last_hidden_state
                out[start:start+len(t)] = h.cpu().numpy()
                peak = max(peak, current_rss())
                if peak > MAX_RSS_BYTES:
                    raise RuntimeError("PILOT_BUDGET_BLOCKED: BERT cache RAM >14 GiB")
            out.flush()
    elapsed = time.perf_counter() - t0
    meta = {"tokens_sha256": tokens_sha, "bert_weights_sha256": EXPECTED["bert_weights_sha256"],
            "elapsed_seconds": elapsed, "peak_rss_bytes": peak, "batch_size": BATCH_BERT,
            "thread_count": torch.get_num_threads(), "reused": False,
            "paths": {k: str(v) for k, v in paths.items()}}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if elapsed > 1800:
        raise RuntimeError("PILOT_BUDGET_BLOCKED: initial BERT cache >30 minutes")
    return meta


class PilotHead(nn.Module):
    def __init__(self, level: str):
        super().__init__()
        assert level in ("B0", "B1", "C1")
        self.level = level
        self.proj = nn.ModuleList([nn.Linear(768, 128), nn.Linear(74, 128), nn.Linear(35, 128)])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(50, 128) * 0.02) for _ in MODS])
        if level != "B0":
            self.p_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in MODS])
            self.o_embed = nn.ModuleList([nn.Embedding(3, 128) for _ in MODS])
        if level == "C1":
            self.gates = nn.ModuleList([nn.Sequential(nn.Linear(134, 32), nn.GELU(), nn.Linear(32, 1))
                                        for _ in MODS])
        self.fuse = nn.Sequential(nn.Linear(384, 128), nn.GELU())
        self.cls = nn.Linear(128, 3)
        self.reg = nn.Linear(128, 1)

    @staticmethod
    def pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float().unsqueeze(-1)
        return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, text_state, audio, vision, p, o, unknown_excluded=False):
        xs = (text_state, audio, vision)
        hs = []
        fractions = []
        for m in range(3):
            base = self.proj[m](xs[m])
            pos = self.pos[m].unsqueeze(0)
            if self.level == "B0":
                z = F.gelu(base + pos)
                q = p[:, m] != 0
                h = self.pool(z, q)
            else:
                state = self.p_embed[m](p[:, m].long()) + self.o_embed[m](o[:, m].long()) + pos
                z = F.gelu(base + state)
                q = ((p[:, m] == 1) & (o[:, m] == 1)) if unknown_excluded else ((p[:, m] != 0) & (o[:, m] != 0))
                content = self.pool(z, q)
                state_summary = self.pool(F.gelu(state), p[:, m] != 0)
                h = content + state_summary
            hs.append(h)
            if self.level == "C1":
                fractions.append(torch.stack([(p[:, m] == k).float().mean(dim=1) for k in range(3)] +
                                             [(o[:, m] == k).float().mean(dim=1) for k in range(3)], dim=1))
        alpha = None
        if self.level == "C1":
            r = torch.cat([self.gates[m](torch.cat([hs[m], fractions[m]], dim=1)) for m in range(3)], dim=1)
            alpha = torch.softmax(r, dim=1)
            hs = [3 * alpha[:, m:m+1] * hs[m] for m in range(3)]
        h = self.fuse(torch.cat(hs, dim=1))
        return self.cls(h), 3.0 * torch.tanh(self.reg(h)).squeeze(1), alpha


def tensor_inputs(data: dict, text_states: np.ndarray, rows: np.ndarray, entries=None):
    """Return batch tensors. entries parallel rows if synthetically altered."""
    if entries is None:
        t = np.asarray(text_states[rows]).copy()
        a = data["audio"][rows].copy()
        v = data["vision"][rows].copy()
        p = np.stack([data[f"{m}_P"][rows] for m in MODS], axis=1).copy()
        o = np.stack([data[f"{m}_O"][rows] for m in MODS], axis=1).copy()
    else:
        raise NotImplementedError("augmented batches are prepared by augmented_inputs")
    return tuple(torch.from_numpy(x) for x in (t, a, v, p, o))


def targets(data: dict, rows: np.ndarray):
    samples = data["samples"]
    return (torch.tensor([int(samples[i]["classification_label"]) for i in rows], dtype=torch.long),
            torch.tensor([float(samples[i]["regression_label"]) for i in rows], dtype=torch.float32))


def losses(logits, intensity, class_y, rating_y, lambda_reg: float):
    ce = F.cross_entropy(logits, class_y)
    huber = F.huber_loss(intensity, rating_y, delta=1.0)
    return ce + lambda_reg * huber, ce, huber


def metrics_from_arrays(yc, yr, logits, pred_reg):
    pred_cls = logits.argmax(axis=1)
    if len(yr) < 2 or np.std(yr) == 0 or np.std(pred_reg) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(yr, pred_reg)[0, 1])
    return {"n": len(yc), "accuracy": float(accuracy_score(yc, pred_cls)),
            "macro_f1": float(f1_score(yc, pred_cls, average="macro", labels=[0, 1, 2], zero_division=0)),
            "weighted_f1": float(f1_score(yc, pred_cls, average="weighted", labels=[0, 1, 2], zero_division=0)),
            "mae": float(mean_absolute_error(yr, pred_reg)), "pearson": pearson,
            "predicted_class_counts": {str(k): int((pred_cls == k).sum()) for k in range(3)}}


def clean_predict(model: PilotHead, data: dict, states: np.ndarray, batch_size=BATCH_HEAD, unknown_excluded=False):
    model.eval()
    logits, intensity, alphas = [], [], []
    with torch.inference_mode():
        for start in range(0, len(data["samples"]), batch_size):
            rows = np.arange(start, min(len(data["samples"]), start + batch_size))
            inp = tensor_inputs(data, states, rows)
            z, y, alpha = model(*inp, unknown_excluded=unknown_excluded)
            logits.append(z.numpy())
            intensity.append(y.numpy())
            if alpha is not None:
                alphas.append(alpha.numpy())
    return np.concatenate(logits), np.concatenate(intensity), (np.concatenate(alphas) if alphas else None)


def init_run(run_id: str, level: str, seed: int, lambda_reg: float, epochs: int, condition: str):
    run = RUN_DIR / run_id
    run.mkdir(parents=True, exist_ok=False)
    source_mask = mask_path(seed)
    if not source_mask.exists():
        raise RuntimeError(f"mask manifest missing: {source_mask}")
    shutil.copyfile(source_mask, run / "mask_manifest.jsonl")
    config = {"run_id": run_id, "model_level": level, "seed": seed,
              "lambda_cls": 1.0, "lambda_reg": lambda_reg,
              "epochs_cap": epochs, "batch_size": BATCH_HEAD,
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "huber_delta": 1.0, "condition": condition,
              "source_version": "cleaning_2026e_v2/aligned", "bert_frozen": True,
              "unknown_policy": "retained" if level != "B0" else "no_state",
              "mask_sha256": sha256(run / "mask_manifest.jsonl")}
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
           "ram_total_bytes": psutil.virtual_memory().total,
           "code_sha256": sha256(Path(__file__)),
           "v2_manifest_sha256": sha256(ROOT / "manifest.json"),
           "scaler_sha256": EXPECTED["scaler_sha256"],
           "bert_weights_sha256": EXPECTED["bert_weights_sha256"],
           "bert_vocab_sha256": EXPECTED["bert_vocab_sha256"],
           "mask_sha256": config["mask_sha256"],
           "allowed_input_splits": ["train", "valid"]}
    (run / "environment_checksums.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (run / "failures.jsonl").write_text("", encoding="utf-8")
    (run / "training_log.jsonl").write_text("", encoding="utf-8")
    (run / "predictions_valid.jsonl").write_text("", encoding="utf-8")
    return run


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def batch_clean_loss(model, data: dict, states: np.ndarray, lambda_reg: float):
    model.eval()
    total, ce_total, huber_total = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(data["samples"]), BATCH_HEAD):
            rows = np.arange(start, min(len(data["samples"]), start+BATCH_HEAD))
            z, y, _ = model(*tensor_inputs(data, states, rows))
            cy, ry = targets(data, rows)
            loss, ce, huber = losses(z, y, cy, ry, lambda_reg)
            total += float(loss) * len(rows)
            ce_total += float(ce) * len(rows)
            huber_total += float(huber) * len(rows)
    n = len(data["samples"])
    return {"total": total/n, "ce": ce_total/n, "huber": huber_total/n}


def write_clean_predictions(run: Path, valid: dict, z: np.ndarray, y: np.ndarray):
    path = run / "predictions_valid.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(valid["samples"]):
            record = {"split": "valid", "condition": "clean", "row": i,
                      "sample_id": sample["id"],
                      "true_class": int(sample["classification_label"]),
                      "true_intensity": float(sample["regression_label"]),
                      "logits": [float(v) for v in z[i]],
                      "predicted_class": int(z[i].argmax()),
                      "predicted_intensity": float(y[i])}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fit_clean(run_id: str, seed: int, lambda_reg: float, epochs: int, warmup=False, level="B0"):
    set_determinism(seed)
    train, valid = load_split("train"), load_split("valid")
    cache_meta = json.loads((CACHE / "clean_text_cache.json").read_text(encoding="utf-8"))
    train_states = np.load(CACHE / "clean_train.npy", mmap_mode="r")
    valid_states = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    condition = "clean_lambda_warmup" if warmup else ("clean_teacher" if level == "B1" else "clean_baseline")
    run = init_run(run_id, level, seed, lambda_reg, epochs, condition)
    t0 = time.perf_counter()
    peak = current_rss()
    model = PilotHead(level)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    epoch_times = []
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    try:
        for epoch in range(1, epochs+1):
            et = time.perf_counter()
            model.train()
            order = np.random.default_rng(stable_seed("shuffle", run_id, seed, epoch)).permutation(len(train["samples"]))
            accum = np.zeros(3, dtype=float)
            for start in range(0, len(order), BATCH_HEAD):
                rows = order[start:start+BATCH_HEAD]
                inputs = tensor_inputs(train, train_states, rows)
                cy, ry = targets(train, rows)
                optimizer.zero_grad(set_to_none=True)
                z, y, _ = model(*inputs)
                loss, ce, huber = losses(z, y, cy, ry, lambda_reg)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite train loss")
                loss.backward()
                optimizer.step()
                accum += np.array([float(loss.detach()), float(ce.detach()), float(huber.detach())]) * len(rows)
            valid_losses = batch_clean_loss(model, valid, valid_states, lambda_reg)
            z, y, _ = clean_predict(model, valid, valid_states)
            yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
            yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
            met = metrics_from_arrays(yc, yr, z, y)
            epoch_sec = time.perf_counter() - et
            epoch_times.append(epoch_sec)
            peak = max(peak, current_rss())
            append_jsonl(run / "training_log.jsonl",
                         {"epoch": epoch, "train_loss": (accum / len(order)).tolist(),
                          "valid_loss": valid_losses, "valid_metrics": met,
                          "epoch_seconds": epoch_sec, "rss_bytes": peak})
            print(json.dumps({"run": run_id, "epoch": epoch, "seconds": round(epoch_sec, 2),
                              "valid_macro_f1": round(met["macro_f1"], 4),
                              "valid_mae": round(met["mae"], 4)}, ensure_ascii=False), flush=True)
            if valid_losses["total"] < best_loss - 1e-6:
                best_loss = valid_losses["total"]
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), run / "checkpoint.pt")
            else:
                no_improve += 1
            if peak > MAX_RSS_BYTES or (time.perf_counter()-t0+max(epoch_times)*max(0, epochs-epoch)) > MAX_SEED_SECONDS:
                raise RuntimeError("PILOT_BUDGET_BLOCKED: projected single-seed RAM/time cap exceeded")
            if not warmup and no_improve >= 3:
                break
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        z, y, _ = clean_predict(model, valid, valid_states)
        final = metrics_from_arrays(yc, yr, z, y)
        write_clean_predictions(run, valid, z, y)
        result = {"status": "COMPLETE", "run_id": run_id, "level": level,
                  "seed": seed, "lambda_reg": lambda_reg,
                  "clean": final, "best_epoch": best_epoch, "best_valid_loss": best_loss,
                  "epoch_times_seconds": epoch_times,
                  "mean_epoch_seconds": float(np.mean(epoch_times)),
                  "train_wall_seconds": time.perf_counter()-t0,
                  "bert_cache_seconds": cache_meta["elapsed_seconds"],
                  "bert_cache_reused": True, "peak_rss_bytes": peak,
                  "checkpoint_sha256": sha256(run / "checkpoint.pt")}
        (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "error": repr(exc), "elapsed_seconds": time.perf_counter()-t0,
                   "peak_rss_bytes": peak, "completed_epochs": len(epoch_times)}
        append_jsonl(run / "failures.jsonl", failure)
        (run / "metrics.json").write_text(json.dumps({"status": "FAILED", **failure}, indent=2), encoding="utf-8")
        raise


def choose_lambda(results: list[dict]) -> dict:
    metrics = [r["clean"] for r in results]
    best_acc = max(m["accuracy"] for m in metrics)
    pearsons = [m["pearson"] for m in metrics if m["pearson"] is not None]
    best_rho = max(pearsons) if pearsons else None
    accepted = [i for i, m in enumerate(metrics) if m["accuracy"] >= best_acc - 0.02 and
                (best_rho is None or (m["pearson"] is not None and m["pearson"] >= best_rho - 0.03))]
    if not accepted:
        accepted = [1]
    nondominated = [i for i in accepted if not any(
        j != i and metrics[j]["macro_f1"] >= metrics[i]["macro_f1"] and
        metrics[j]["mae"] <= metrics[i]["mae"] and
        (metrics[j]["macro_f1"] > metrics[i]["macro_f1"] or metrics[j]["mae"] < metrics[i]["mae"])
        for j in accepted)]
    chosen_idx = min(nondominated, key=lambda i: (abs(i-1), i))
    return {"chosen_index": chosen_idx, "lambda_reg": results[chosen_idx]["lambda_reg"],
            "accepted_indices": accepted, "nondominated_indices": nondominated,
            "warmup_metrics": [{"run_id": r["run_id"], "lambda_reg": r["lambda_reg"],
                                "clean": r["clean"]} for r in results]}


def run_r0():
    smoke_path = RESULT / "smoke_check.json"
    assert smoke_path.exists() and json.loads(smoke_path.read_text(encoding="utf-8"))["status"] == "PASSED"
    set_determinism(SEEDS[0])
    train, valid = load_split("train"), load_split("valid")
    if not (CACHE / "clean_text_cache.json").exists():
        cache_clean_bert(train, valid)
    states = np.load(CACHE / "clean_train.npy", mmap_mode="r")
    initial = PilotHead("B0")
    rows = np.arange(min(256, len(train["samples"])))
    cy, ry = targets(train, rows)
    with torch.inference_mode():
        z, y, _ = initial(*tensor_inputs(train, states, rows))
        _, ce, huber = losses(z, y, cy, ry, 1.0)
    q = float(ce / huber)
    if not np.isfinite(q) or q <= 0:
        raise RuntimeError("invalid train-only lambda scale")
    warmup = []
    for label, mult in (("low", 0.5), ("middle", 1.0), ("high", 2.0)):
        warmup.append(fit_clean(f"R0_lambda_{label}_seed1729", SEEDS[0], q*mult, 3, warmup=True))
    selection = choose_lambda(warmup)
    selection["initial_train_ce"] = float(ce)
    selection["initial_train_huber"] = float(huber)
    selection["q"] = q
    (RESULT / "lambda_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    result = fit_clean("R0_seed1729", SEEDS[0], selection["lambda_reg"], 12)
    print(json.dumps({"step": "R0_COMPLETE", "selected_lambda": selection["lambda_reg"],
                      "metrics": result["clean"], "mean_epoch_seconds": result["mean_epoch_seconds"],
                      "peak_rss_bytes": result["peak_rss_bytes"]}, ensure_ascii=False), flush=True)
    return result


def read_mask_rows(seed: int) -> list[dict]:
    return [json.loads(line) for line in mask_path(seed).read_text(encoding="utf-8").splitlines()]


def masks_by_phase(seed: int):
    rows = read_mask_rows(seed)
    train = {r["row"]: r for r in rows if r["phase"] == "train"}
    valid = {phase: {scenario: {r["row"]: r for r in rows
                                if r["phase"] == phase and r["scenario"] == scenario}
                     for scenario in SCENARIOS}
             for phase in ("selection", "audit")}
    assert len(train) == SHAPES["train"]
    assert all(len(valid[phase][s]) == SHAPES["valid"] for phase in valid for s in SCENARIOS)
    return train, valid


def gap_cache_key(entry: dict) -> str:
    return f"{entry['split']}|{entry['phase']}|{entry['scenario']}|{entry['row']}"


def cache_gap_bert(train: dict, valid: dict, seed: int) -> dict:
    """Encode only genuinely altered text token sequences, after [UNK] intervention."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"gap_text_seed{seed}.npy"
    meta_path = CACHE / f"gap_text_seed{seed}.json"
    mask_sha = sha256(mask_path(seed))
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["mask_sha256"] == mask_sha
        assert meta["bert_weights_sha256"] == EXPECTED["bert_weights_sha256"]
        assert np.load(path, mmap_mode="r").shape == (meta["unique_sequences"], 50, 768)
        return {**meta, "reused": True, "path": str(path)}
    rows = read_mask_rows(seed)
    unique = {}
    tokens = []
    lookup = {}
    for entry in rows:
        if not entry["injectable"] or "T" not in entry["modalities"]:
            continue
        data = train if entry["split"] == "train" else valid
        token = data["text_bert"][entry["row"]].copy()
        a, b = entry["spans"]["T"]
        token[0, a:b] = 100
        key = token.tobytes()
        if key not in unique:
            unique[key] = len(tokens)
            tokens.append(token)
        lookup[gap_cache_key(entry)] = unique[key]
    if not tokens:
        raise RuntimeError("no text gap sequences")
    from transformers import BertModel
    set_determinism(seed)
    t0 = time.perf_counter()
    model = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(len(tokens), 50, 768))
    peak = current_rss()
    with torch.inference_mode():
        for start in range(0, len(tokens), BATCH_BERT):
            t = torch.from_numpy(np.stack(tokens[start:start+BATCH_BERT]))
            h = model(input_ids=t[:, 0], attention_mask=t[:, 1],
                      token_type_ids=t[:, 2]).last_hidden_state
            out[start:start+len(t)] = h.numpy()
            peak = max(peak, current_rss())
            if peak > MAX_RSS_BYTES:
                raise RuntimeError("PILOT_BUDGET_BLOCKED: text-gap cache RAM >14 GiB")
        out.flush()
    elapsed = time.perf_counter()-t0
    meta = {"seed": seed, "mask_sha256": mask_sha,
            "bert_weights_sha256": EXPECTED["bert_weights_sha256"],
            "unique_sequences": len(tokens), "lookup": lookup,
            "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
            "reused": False, "path": str(path)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if elapsed > 1800:
        raise RuntimeError("PILOT_BUDGET_BLOCKED: text-gap cache >30 minutes")
    return meta


def augmented_inputs(data: dict, clean_states: np.ndarray, gap_states: np.ndarray,
                     lookup: dict, specs: list[tuple[int, dict]]):
    t, a, v, p, o = [], [], [], [], []
    for row, entry in specs:
        if entry is None:
            t.append(np.asarray(clean_states[row]))
            a.append(data["audio"][row])
            v.append(data["vision"][row])
            p.append(np.stack([data[f"{m}_P"][row] for m in MODS]))
            o.append(np.stack([data[f"{m}_O"][row] for m in MODS]))
            continue
        item = apply_mask(data, row, entry)
        key = gap_cache_key(entry)
        t.append(np.asarray(gap_states[lookup[key]] if key in lookup else clean_states[row]))
        a.append(item["audio"])
        v.append(item["vision"])
        p.append(np.stack([item[f"{m}_P"] for m in MODS]))
        o.append(np.stack([item[f"{m}_O"] for m in MODS]))
    return tuple(torch.from_numpy(np.stack(x).copy()) for x in (t, a, v, p, o))


def predict_specs(model, data: dict, clean_states: np.ndarray, gap_states: np.ndarray,
                  lookup: dict, entries, unknown_excluded=False):
    model.eval()
    logits, intensity, alphas = [], [], []
    with torch.inference_mode():
        for start in range(0, len(data["samples"]), BATCH_HEAD):
            rows = range(start, min(len(data["samples"]), start+BATCH_HEAD))
            specs = [(i, entries[i] if entries is not None else None) for i in rows]
            z, y, alpha = model(*augmented_inputs(data, clean_states, gap_states, lookup, specs),
                                unknown_excluded=unknown_excluded)
            logits.append(z.numpy())
            intensity.append(y.numpy())
            if alpha is not None:
                alphas.append(alpha.numpy())
    return np.concatenate(logits), np.concatenate(intensity), (np.concatenate(alphas) if alphas else None)


def scenario_metrics(valid: dict, clean_z: np.ndarray, clean_y: np.ndarray,
                     gap_z: np.ndarray, gap_y: np.ndarray, entries: dict):
    eligible = np.array([entries[i]["injectable"] for i in range(len(valid["samples"]))], dtype=bool)
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    if not eligible.any():
        return {"n_eligible": 0, "n_not_injectable": len(eligible), "status": "NO_ELIGIBLE_SAMPLES"}
    clean = metrics_from_arrays(yc[eligible], yr[eligible], clean_z[eligible], clean_y[eligible])
    gap = metrics_from_arrays(yc[eligible], yr[eligible], gap_z[eligible], gap_y[eligible])
    delta = {key: (gap[key] - clean[key]) if gap[key] is not None and clean[key] is not None else None
             for key in ("accuracy", "macro_f1", "mae", "pearson")}
    return {"n_eligible": int(eligible.sum()), "n_not_injectable": int((~eligible).sum()),
            "clean_paired": clean, "gap_paired": gap, "delta": delta}


def evaluate_all(model, valid: dict, clean_states: np.ndarray, gap_states: np.ndarray,
                 lookup: dict, phase_entries: dict, unknown_excluded=False):
    clean_z, clean_y, clean_alpha = predict_specs(model, valid, clean_states, gap_states,
                                                  lookup, None, unknown_excluded)
    yc = np.array([int(s["classification_label"]) for s in valid["samples"]])
    yr = np.array([float(s["regression_label"]) for s in valid["samples"]])
    out = {"clean": metrics_from_arrays(yc, yr, clean_z, clean_y), "scenarios": {}}
    predictions = {"clean": (clean_z, clean_y, clean_alpha)}
    for scenario, entries in phase_entries.items():
        z, y, alpha = predict_specs(model, valid, clean_states, gap_states, lookup,
                                    entries, unknown_excluded)
        out["scenarios"][scenario] = scenario_metrics(valid, clean_z, clean_y, z, y, entries)
        predictions[scenario] = (z, y, alpha)
    return out, predictions


def selection_loss(predictions: dict, valid: dict, lambda_reg: float):
    rows = np.arange(len(valid["samples"]))
    cy, ry = targets(valid, rows)
    values = {}
    for scenario, (z, y, _) in predictions.items():
        loss, ce, huber = losses(torch.from_numpy(z), torch.from_numpy(y), cy, ry, lambda_reg)
        values[scenario] = {"total": float(loss), "ce": float(ce), "huber": float(huber)}
    return 0.5 * values["clean"]["total"] + 0.5 * float(np.mean(
        [values[s]["total"] for s in SCENARIOS])), values


def write_all_predictions(run: Path, valid: dict, predictions: dict):
    with (run / "predictions_valid.jsonl").open("w", encoding="utf-8") as f:
        for scenario, (z, y, alpha) in predictions.items():
            for i, sample in enumerate(valid["samples"]):
                record = {"split": "valid", "condition": scenario, "row": i,
                          "sample_id": sample["id"],
                          "true_class": int(sample["classification_label"]),
                          "true_intensity": float(sample["regression_label"]),
                          "logits": [float(x) for x in z[i]],
                          "predicted_class": int(z[i].argmax()),
                          "predicted_intensity": float(y[i])}
                if alpha is not None:
                    record["alpha"] = [float(x) for x in alpha[i]]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def alpha_diagnostics(predictions: dict):
    out = {}
    for scenario, (_, _, alpha) in predictions.items():
        if alpha is None:
            continue
        entropy = -(alpha * np.log(np.clip(alpha, 1e-12, 1))).sum(axis=1)
        out[scenario] = {"mean": alpha.mean(axis=0).tolist(),
                         "std": alpha.std(axis=0).tolist(),
                         "min": alpha.min(axis=0).tolist(),
                         "max": alpha.max(axis=0).tolist(),
                         "entropy_mean": float(entropy.mean()),
                         "collapse_fraction_any_gt_0_95": float((alpha.max(axis=1) > 0.95).mean()),
                         "dominant_counts": {str(m): int((alpha.argmax(axis=1) == m).sum()) for m in range(3)}}
    return out


def fit_augmented(run_id: str, level: str, seed: int, lambda_reg: float, unknown_excluded=False):
    assert level in ("B0", "B1", "C1")
    set_determinism(seed)
    train, valid = load_split("train"), load_split("valid")
    train_entries, valid_entries = masks_by_phase(seed)
    gap_meta = cache_gap_bert(train, valid, seed)
    lookup = gap_meta["lookup"]
    train_clean = np.load(CACHE / "clean_train.npy", mmap_mode="r")
    valid_clean = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")
    run = init_run(run_id, level, seed, lambda_reg, 12, "clean_plus_one_gap_copy")
    t0 = time.perf_counter()
    peak = current_rss()
    model = PilotHead(level)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n = len(train["samples"])
    specs = [(i, None) for i in range(n)] + [(i, train_entries[i]) for i in range(n)]
    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    epoch_times = []
    try:
        for epoch in range(1, 13):
            et = time.perf_counter()
            model.train()
            order = np.random.default_rng(stable_seed("shuffle", run_id, seed, epoch)).permutation(len(specs))
            accum = np.zeros(3, dtype=float)
            for start in range(0, len(order), BATCH_HEAD):
                chosen = [specs[int(i)] for i in order[start:start+BATCH_HEAD]]
                inputs = augmented_inputs(train, train_clean, gap_states, lookup, chosen)
                rows = np.array([i for i, _ in chosen])
                cy, ry = targets(train, rows)
                optimizer.zero_grad(set_to_none=True)
                z, y, _ = model(*inputs, unknown_excluded=unknown_excluded)
                loss, ce, huber = losses(z, y, cy, ry, lambda_reg)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite train loss")
                loss.backward()
                optimizer.step()
                accum += np.array([float(loss.detach()), float(ce.detach()), float(huber.detach())]) * len(chosen)
            selection, pred = evaluate_all(model, valid, valid_clean, gap_states,
                                           lookup, valid_entries["selection"], unknown_excluded)
            score, loss_detail = selection_loss(pred, valid, lambda_reg)
            epoch_sec = time.perf_counter()-et
            epoch_times.append(epoch_sec)
            peak = max(peak, current_rss())
            append_jsonl(run / "training_log.jsonl",
                         {"epoch": epoch, "train_loss": (accum / len(specs)).tolist(),
                          "valid_selection_loss": score, "valid_loss_detail": loss_detail,
                          "valid_selection_clean": selection["clean"],
                          "valid_selection_scenarios": selection["scenarios"],
                          "epoch_seconds": epoch_sec, "rss_bytes": peak})
            print(json.dumps({"run": run_id, "epoch": epoch, "seconds": round(epoch_sec, 2),
                              "selection_loss": round(score, 4),
                              "clean_macro_f1": round(selection["clean"]["macro_f1"], 4),
                              "clean_mae": round(selection["clean"]["mae"], 4)}, ensure_ascii=False), flush=True)
            if score < best_loss - 1e-6:
                best_loss, best_epoch, no_improve = score, epoch, 0
                torch.save(model.state_dict(), run / "checkpoint.pt")
            else:
                no_improve += 1
            if peak > MAX_RSS_BYTES or (time.perf_counter()-t0+max(epoch_times)*max(0, 12-epoch)) > MAX_SEED_SECONDS:
                raise RuntimeError("PILOT_BUDGET_BLOCKED: projected single-seed RAM/time cap exceeded")
            if no_improve >= 3:
                break
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        audit, predictions = evaluate_all(model, valid, valid_clean, gap_states,
                                          lookup, valid_entries["audit"], unknown_excluded)
        write_all_predictions(run, valid, predictions)
        result = {"status": "COMPLETE", "run_id": run_id, "level": level,
                  "seed": seed, "lambda_reg": lambda_reg, "audit": audit,
                  "best_epoch": best_epoch, "best_valid_selection_loss": best_loss,
                  "epoch_times_seconds": epoch_times,
                  "mean_epoch_seconds": float(np.mean(epoch_times)),
                  "train_wall_seconds": time.perf_counter()-t0,
                  "bert_gap_cache_seconds": gap_meta["elapsed_seconds"],
                  "bert_gap_cache_reused": gap_meta["reused"],
                  "peak_rss_bytes": peak, "checkpoint_sha256": sha256(run / "checkpoint.pt")}
        if level == "C1":
            result["alpha_diagnostics"] = alpha_diagnostics(predictions)
        (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        failure = {"run_id": run_id, "error": repr(exc), "elapsed_seconds": time.perf_counter()-t0,
                   "peak_rss_bytes": peak, "completed_epochs": len(epoch_times)}
        append_jsonl(run / "failures.jsonl", failure)
        (run / "metrics.json").write_text(json.dumps({"status": "FAILED", **failure}, indent=2), encoding="utf-8")
        raise


def evaluate_r1x(seed: int, lambda_reg: float):
    set_determinism(seed)
    train, valid = load_split("train"), load_split("valid")
    _, valid_entries = masks_by_phase(seed)
    gap_meta = cache_gap_bert(train, valid, seed)
    run_id = f"R1x_seed{seed}"
    run = init_run(run_id, "B1", seed, lambda_reg, 0, "fixed_R1_weights_unknown_excluded_inference")
    source = RUN_DIR / f"R1_seed{seed}" / "checkpoint.pt"
    model = PilotHead("B1")
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
    clean_states = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")
    start = time.perf_counter()
    audit, predictions = evaluate_all(model, valid, clean_states, gap_states,
                                      gap_meta["lookup"], valid_entries["audit"], True)
    write_all_predictions(run, valid, predictions)
    result = {"status": "COMPLETE", "run_id": run_id, "level": "B1_fixed_weight_sensitivity",
              "source_checkpoint": str(source), "source_checkpoint_sha256": sha256(source),
              "seed": seed, "lambda_reg": lambda_reg, "audit": audit,
              "inference_wall_seconds": time.perf_counter()-start,
              "peak_rss_bytes": current_rss(), "fitted_weights_changed": False}
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def c1_gate_interventions(seed: int):
    """Inference-only gate responsiveness; weights are unchanged."""
    set_determinism(seed)
    valid = load_split("valid")
    clean_states = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    run = RUN_DIR / f"R2_seed{seed}"
    model = PilotHead("C1")
    model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
    _, _, base_alpha = clean_predict(model, valid, clean_states)
    result = {"n": len(valid["samples"]), "baseline_alpha_mean": base_alpha.mean(axis=0).tolist(),
              "interventions": {}}
    from transformers import BertModel
    bert = None
    for mi, mod in enumerate(MODS):
        altered_alpha = []
        if mod == "text":
            bert = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
        with torch.inference_mode():
            for start in range(0, len(valid["samples"]), BATCH_BERT if mod == "text" else BATCH_HEAD):
                rows = np.arange(start, min(len(valid["samples"]), start +
                                               (BATCH_BERT if mod == "text" else BATCH_HEAD)))
                t, a, v, p, o = tensor_inputs(valid, clean_states, rows)
                content = [t, a, v]
                valid_positions = (p[:, mi] == 1) & (o[:, mi] == 1)
                if mod == "text":
                    tokens = torch.from_numpy(valid["text_bert"][rows].copy())
                    ids = tokens[:, 0]
                    valid_positions &= (ids != 101) & (ids != 102) & (ids != 0)
                    ids[valid_positions] = 100
                    t = bert(input_ids=ids, attention_mask=tokens[:, 1],
                             token_type_ids=tokens[:, 2]).last_hidden_state
                    content[0] = t
                else:
                    content[mi][valid_positions] = 0.0
                o[:, mi][valid_positions] = 0
                _, _, alpha = model(content[0], content[1], content[2], p, o)
                altered_alpha.append(alpha.numpy())
        changed = np.concatenate(altered_alpha)
        diff = changed - base_alpha
        result["interventions"][mod] = {
            "altered_alpha_mean": changed.mean(axis=0).tolist(),
            "mean_target_alpha_change": float(diff[:, mi].mean()),
            "median_target_alpha_change": float(np.median(diff[:, mi])),
            "fraction_target_alpha_decreased": float((diff[:, mi] < 0).mean()),
            "mean_absolute_alpha_change": float(np.abs(diff).mean()),
            "max_absolute_alpha_change": float(np.abs(diff).max()),
        }
    (run / "gate_interventions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["gate_interventions"] = result
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result


def evaluate_r0_gaps(seed: int):
    set_determinism(seed)
    train, valid = load_split("train"), load_split("valid")
    _, valid_entries = masks_by_phase(seed)
    gap_meta = cache_gap_bert(train, valid, seed)
    model = PilotHead("B0")
    run = RUN_DIR / f"R0_seed{seed}"
    model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
    clean_states = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    gap_states = np.load(CACHE / f"gap_text_seed{seed}.npy", mmap_mode="r")
    audit, predictions = evaluate_all(model, valid, clean_states, gap_states,
                                      gap_meta["lookup"], valid_entries["audit"])
    write_all_predictions(run, valid, predictions)
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["audit"] = audit
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "cache_clean", "r0", "r0_seed", "r0_gaps", "r0a", "r1", "r1x", "c1", "c1_gate", "teacher"])
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit("seed must be one of the frozen seeds")
    if args.command == "smoke":
        smoke()
    elif args.command == "cache_clean":
        check = RESULT / "smoke_check.json"
        assert check.exists() and json.loads(check.read_text(encoding="utf-8"))["status"] == "PASSED"
        set_determinism(SEEDS[0])
        train, valid = load_split("train"), load_split("valid")
        result = cache_clean_bert(train, valid)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "r0":
        if args.seed != SEEDS[0]:
            raise SystemExit("r0 lambda selection is first seed only; use r0_seed")
        run_r0()
    elif args.command == "r0_seed":
        train, valid = load_split("train"), load_split("valid")
        build_masks(train, valid, args.seed)
        selection = json.loads((RESULT / "lambda_selection.json").read_text(encoding="utf-8"))
        result = fit_clean(f"R0_seed{args.seed}", args.seed, selection["lambda_reg"], 12)
        print(json.dumps({"step": "R0_SEED_COMPLETE", "seed": args.seed,
                          "metrics": result["clean"]}, ensure_ascii=False))
    elif args.command == "r0_gaps":
        audit = evaluate_r0_gaps(args.seed)
        print(json.dumps({"step": "R0_GAP_AUDIT", "clean": audit["clean"],
                          "scenarios": {k: v["gap_paired"] for k, v in audit["scenarios"].items()}}, ensure_ascii=False))
    elif args.command in ("r0a", "r1", "r1x", "c1"):
        selection = json.loads((RESULT / "lambda_selection.json").read_text(encoding="utf-8"))
        lam = selection["lambda_reg"]
        seed = args.seed
        if not mask_path(seed).exists():
            train, valid = load_split("train"), load_split("valid")
            build_masks(train, valid, seed)
        if args.command == "r0a":
            fit_augmented(f"R0a_seed{seed}", "B0", seed, lam)
        elif args.command == "r1":
            fit_augmented(f"R1_seed{seed}", "B1", seed, lam)
        elif args.command == "r1x":
            evaluate_r1x(seed, lam)
        elif args.command == "c1":
            fit_augmented(f"R2_seed{seed}", "C1", seed, lam)
    elif args.command == "c1_gate":
        result = c1_gate_interventions(args.seed)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "teacher":
        selection = json.loads((RESULT / "lambda_selection.json").read_text(encoding="utf-8"))
        if not mask_path(args.seed).exists():
            train, valid = load_split("train"), load_split("valid")
            build_masks(train, valid, args.seed)
        result = fit_clean(f"C2_teacher_seed{args.seed}", args.seed, selection["lambda_reg"], 12, level="B1")
        print(json.dumps({"step": "C2_TEACHER_FIRST_SEED", "metrics": result["clean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
