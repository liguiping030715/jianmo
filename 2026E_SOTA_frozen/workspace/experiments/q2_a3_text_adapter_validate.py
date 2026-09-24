"""Validate the future unaligned A3 text adapter using A2 train/valid only."""
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
from q2_bridge import NativeMulT
from q2_unaligned_text_adapter import FrozenUnalignedTextAdapter, file_sha256

DATA = Path("workspace/data/processed/cleaning_2026e_v2/unaligned")
CACHE = Path("workspace/experiments/pilot_cache")
RUNS = Path("workspace/experiments/q2_bridge_runs")
OUT = Path("workspace/results/q2_bridge")
SEEDS = (1729, 2718, 31415)
RTOL = 1e-5
ATOL = 1e-6
FIELDS = ("input_ids", "attention_mask", "token_type_ids")
EXECUTED_M0_SHA256 = "dc0367156de2bd7108d7910557692c173c35964b5c2307beb0dd8d32a2fedccf"


def split_text(split):
    if split not in ("train", "valid"):
        raise RuntimeError("forbidden split")
    base = DATA / split
    samples = json.loads((base / "samples.json").read_text(encoding="utf-8"))
    stored = np.load(base / "text_bert.npy", allow_pickle=False)
    expected_n = 3395 if split == "train" else 728
    if len(samples) != expected_n or stored.shape != (expected_n, 3, 50):
        raise RuntimeError(f"A2 {split} text shape/count mismatch")
    return samples, stored


def compare_tokens(split, samples, stored, generated):
    diff = stored != generated
    mismatched = np.flatnonzero(np.any(diff, axis=(1, 2)))
    first = None
    if len(mismatched):
        row = int(mismatched[0])
        field, position = np.argwhere(diff[row])[0]
        first = {"row": row, "sample_id": samples[row]["id"],
                 "field": FIELDS[int(field)], "position": int(position),
                 "stored": int(stored[row, field, position]),
                 "regenerated": int(generated[row, field, position])}
    return {"split": split, "exact_rows": len(samples)-len(mismatched),
            "total_rows": len(samples), "mismatched_rows": int(len(mismatched)),
            "first_mismatch": first,
            "attention_mask_exact": bool(np.array_equal(stored[:, 1], generated[:, 1])),
            "token_type_ids_exact": bool(np.array_equal(stored[:, 2], generated[:, 2]))}


def saved_clean(run, samples):
    rows = []
    for line in (run / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["condition"] == "clean":
            rows.append(rec)
    if len(rows) != 728 or any(rec["row"] != i or rec["sample_id"] != samples[i]["id"]
                               for i, rec in enumerate(rows)):
        raise RuntimeError("saved Bridge clean prediction order mismatch")
    return (np.array([r["logits"] for r in rows]),
            np.array([r["predicted_intensity"] for r in rows]))


def metric_differences(a, b):
    return {key: (None if a[key] is None or b[key] is None else float(b[key] - a[key]))
            for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}


def validate():
    if file_sha256(Path("q2_bridge.py")) != EXECUTED_M0_SHA256:
        raise RuntimeError("M0 source differs from executed checkpoints")
    adapter = FrozenUnalignedTextAdapter()
    train_samples, stored_train = split_text("train")
    valid_samples, stored_valid = split_text("valid")
    generated_train = adapter.tokenize([s["raw_text"] for s in train_samples])
    generated_valid_bundle = adapter.encode_raw_text([s["raw_text"] for s in valid_samples])
    generated_valid = generated_valid_bundle["text_bert"]
    tokens = {"train": compare_tokens("train", train_samples, stored_train, generated_train),
              "valid": compare_tokens("valid", valid_samples, stored_valid, generated_valid)}
    cache_meta = json.loads((CACHE / "clean_text_cache.json").read_text(encoding="utf-8"))
    if cache_meta["bert_weights_sha256"] != adapter.provenance()["file_sha256"]["pytorch_model.bin"]:
        raise RuntimeError("Bridge BERT cache model provenance mismatch")
    if cache_meta["batch_size"] != 16 or cache_meta["thread_count"] != 8:
        raise RuntimeError("Bridge BERT cache execution settings mismatch")
    for split, stored in (("train", stored_train), ("valid", stored_valid)):
        if hashlib.sha256(stored.tobytes()).hexdigest() != cache_meta["tokens_sha256"][split]:
            raise RuntimeError(f"Bridge {split} token cache provenance mismatch")
    bridge_text = np.load(CACHE / "clean_valid.npy", mmap_mode="r")
    generated_hidden = generated_valid_bundle["text_hidden"]
    if bridge_text.shape != (728, 50, 768) or generated_hidden.shape != (728, 50, 768):
        raise RuntimeError("BERT hidden-state shape mismatch")
    diff = generated_hidden.astype(np.float64) - np.asarray(bridge_text, dtype=np.float64)
    hidden = {"shape": [728, 50, 768],
              "bridge_dtype": str(bridge_text.dtype),
              "adapter_dtype": str(generated_hidden.dtype),
              "bridge_finite": bool(np.isfinite(bridge_text).all()),
              "adapter_finite": bool(np.isfinite(generated_hidden).all()),
              "max_abs_error": float(np.max(np.abs(diff))),
              "mean_abs_error": float(np.mean(np.abs(diff))),
              "rmse": float(np.sqrt(np.mean(diff ** 2))),
              "allclose": bool(np.allclose(generated_hidden, bridge_text, rtol=RTOL, atol=ATOL)),
              "rtol": RTOL, "atol": ATOL,
              "attention_mask_exact": tokens["valid"]["attention_mask_exact"]}
    valid_data = ru.load_split("valid")
    if [s["id"] for s in valid_data["samples"]] != [s["id"] for s in valid_samples]:
        raise RuntimeError("valid sample order differs")
    p_from_attention = generated_valid_bundle["text_P"]
    o_from_attention = generated_valid_bundle["text_O"]
    mask = {"text_P_equals_regenerated_attention_mask": bool(np.array_equal(p_from_attention, generated_valid[:, 1])),
            "text_P_equals_stored_A2_P": bool(np.array_equal(p_from_attention, valid_data["text_P"])),
            "text_O_equals_stored_A2_O": bool(np.array_equal(o_from_attention, valid_data["text_O"])),
            "no_hidden_magnitude_missingness_rule": True,
            "M0_text_pool_criterion": "P != 0; O unused by M0",
            "padded_hidden_states_enter_final_pool": False}
    path_b_data = dict(valid_data)
    path_b_data["text_P"] = p_from_attention
    path_b_data["text_O"] = o_from_attention
    yc = np.array([int(s["classification_label"]) for s in valid_samples], dtype=np.int64)
    yr = np.array([float(s["regression_label"]) for s in valid_samples])
    per_seed = {}
    csv_rows = []
    for seed in SEEDS:
        run = RUNS / f"M0_seed{seed}"
        env = json.loads((run / "environment_checksums.json").read_text(encoding="utf-8"))
        saved_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if env["code_sha256"] != EXECUTED_M0_SHA256 or saved_metrics["status"] != "COMPLETE":
            raise RuntimeError(f"seed {seed} checkpoint provenance mismatch")
        if file_sha256(run / "checkpoint.pt") != saved_metrics["checkpoint_sha256"]:
            raise RuntimeError(f"seed {seed} checkpoint hash mismatch")
        model = NativeMulT(False)
        model.load_state_dict(torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True))
        model.eval()
        path_a_z, path_a_y = ru.predict_all(model, valid_data, bridge_text, None, {}, None)
        path_b_z, path_b_y = ru.predict_all(model, path_b_data, generated_hidden, None, {}, None)
        old_z, old_y = saved_clean(run, valid_samples)
        if max(float(np.max(np.abs(path_a_z-old_z))), float(np.max(np.abs(path_a_y-old_y)))) > 1e-5:
            raise RuntimeError(f"seed {seed} cache PATH A differs from saved clean predictions")
        metrics_a = ru.metrics_from_arrays(yc, yr, path_a_z, path_a_y)
        metrics_b = ru.metrics_from_arrays(yc, yr, path_b_z, path_b_y)
        for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson"):
            if abs(metrics_a[key] - saved_metrics["audit"]["clean"][key]) > 1e-6:
                raise RuntimeError(f"seed {seed} PATH A metric mismatch: {key}")
        dz = np.abs(path_a_z - path_b_z)
        dy = np.abs(path_a_y - path_b_y)
        pred_a, pred_b = path_a_z.argmax(axis=1), path_b_z.argmax(axis=1)
        comparison = {
            "checkpoint_sha256": saved_metrics["checkpoint_sha256"],
            "max_abs_logit_difference": float(dz.max()),
            "mean_abs_logit_difference": float(dz.mean()),
            "classification_disagreement_count": int(np.count_nonzero(pred_a != pred_b)),
            "max_abs_regression_difference": float(dy.max()),
            "mean_abs_regression_difference": float(dy.mean()),
            "logits_allclose": bool(np.allclose(path_a_z, path_b_z, rtol=RTOL, atol=ATOL)),
            "regression_allclose": bool(np.allclose(path_a_y, path_b_y, rtol=RTOL, atol=ATOL)),
            "metrics_path_A": metrics_a,
            "metrics_path_B": metrics_b,
            "metric_difference_B_minus_A": metric_differences(metrics_a, metrics_b)}
        per_seed[str(seed)] = comparison
        csv_rows.append({"seed": seed,
                         "max_abs_logit_difference": comparison["max_abs_logit_difference"],
                         "mean_abs_logit_difference": comparison["mean_abs_logit_difference"],
                         "classification_disagreement_count": comparison["classification_disagreement_count"],
                         "max_abs_regression_difference": comparison["max_abs_regression_difference"],
                         "mean_abs_regression_difference": comparison["mean_abs_regression_difference"],
                         **{f"A_{k}": metrics_a[k] for k in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")},
                         **{f"B_{k}": metrics_b[k] for k in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")}})
        print(json.dumps({"seed": seed, "max_logit_difference": comparison["max_abs_logit_difference"],
                          "class_disagreement": comparison["classification_disagreement_count"]}), flush=True)
    exact_tokens = all(tokens[s]["mismatched_rows"] == 0 for s in ("train", "valid"))
    equivalent = (exact_tokens and hidden["allclose"] and hidden["bridge_finite"] and hidden["adapter_finite"]
                  and all(mask[k] for k in ("text_P_equals_regenerated_attention_mask",
                                             "text_P_equals_stored_A2_P", "text_O_equals_stored_A2_O"))
                  and all(rec["logits_allclose"] and rec["regression_allclose"] and
                          rec["classification_disagreement_count"] == 0 and
                          all(abs(value) <= 1e-6 for value in rec["metric_difference_B_minus_A"].values())
                          for rec in per_seed.values()))
    status = "BRIDGE_A3_TEXT_ADAPTER_VALIDATED" if equivalent else "BRIDGE_A3_TEXT_ADAPTER_MISMATCH"
    result = {"status": status,
              "scope": "interface validation using only A2 train/valid and frozen M0 checkpoints; no A3 sample access",
              "adapter_source": "q2_unaligned_text_adapter.py",
              "adapter_source_sha256": file_sha256(Path("q2_unaligned_text_adapter.py")),
              "M0_source_sha256": EXECUTED_M0_SHA256,
              "provenance": adapter.provenance(),
              "token_equivalence": tokens,
              "hidden_state_equivalence": hidden,
              "mask_semantics": mask,
              "prediction_equivalence": per_seed,
              "future_unaligned_A3_runner_contract": {
                  "input_fields": ["external official file/order key", "raw_text", "audio [500,74]", "vision [500,35]", "audited audio/vision P/O states"],
                  "text_path": "raw_text -> this pinned adapter -> [3,50] token triplet + [50,768] hidden states; attention_mask -> text P and tokenizer padding O",
                  "model_path": "text hidden/audio/vision plus P/O -> frozen M0 -> three polarity logits/class and continuous intensity",
                  "internal_sample_id_required_by_model": False,
                  "correspondence": "caller retains external official file/order key; no synthetic ID needed",
                  "version_rule": "do not mix aligned A3 text_bert with unaligned A/V",
                  "validation_limit": "contract only; no A3 runner or A3 sample was executed"},
              "accessed_competition_splits": ["A2 train", "A2 valid"],
              "csv_rows": csv_rows}
    return result, csv_rows


def write_outputs(result, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_text_adapter_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (OUT / "a3_text_adapter_equivalence.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if result["status"] == "BRIDGE_A3_TEXT_ADAPTER_BLOCKED":
        lines = ["# Q2 A3 text adapter validation", "", "**Blocked before completing validation.**",
                 "", f"Error: `{result['error']}`.", "",
                 "No training or A3 sample access occurred.", ""]
    else:
        token = result["token_equivalence"]
        hidden = result["hidden_state_equivalence"]
        prov = result["provenance"]
        lines = ["# Q2 A3 text adapter validation", "",
                 f"**Status: `{result['status']}`.** This is an A2-only interface validation for a future unaligned A3 runner. No model was trained, changed or tuned; no A2 test, A3 sample or A4 was accessed.", "",
                 "## Reusable adapter and pinned settings", "",
                 "[FrozenUnalignedTextAdapter](../../../q2_unaligned_text_adapter.py) implements `raw_text → BertTokenizerFast → [input_ids, attention_mask, token_type_ids] → frozen BertModel → 50×768 float32`. It loads only the pinned local `bert-base-uncased`, runs in eval/inference mode with no parameter gradients, and uses BERT batch size 16 and 8 CPU threads, matching the Bridge clean cache.", "",
                 f"Tokenizer: `add_special_tokens=True`, `padding='max_length'`, `truncation=True`, `max_length=50`, right padding/truncation, lower case, explicit attention mask and token-type IDs. Special IDs: CLS 101, SEP 102, PAD 0, UNK 100. Triplet row order: `{list(FIELDS)}`. `transformers=={prov['transformers_version']}`. Model revision `{prov['model_revision']}`.", "",
                 "| Frozen local file | SHA-256 |", "|---|---|"]
        for name, value in prov["file_sha256"].items():
            lines.append(f"| `{name}` | `{value}` |")
        lines += ["", "## Token-level equivalence", "",
                  "| Split | Exact rows | Total | Mismatched rows | First mismatch |",
                  "|---|---:|---:|---:|---|"]
        for split in ("train", "valid"):
            rec = token[split]
            lines.append(f"| {split} | {rec['exact_rows']} | {rec['total_rows']} | {rec['mismatched_rows']} | {rec['first_mismatch'] if rec['first_mismatch'] else 'none'} |")
        lines += ["", "## BERT hidden-state equivalence on every A2 valid row", "",
                  f"Shape {hidden['shape']}; cache dtype `{hidden['bridge_dtype']}`, adapter dtype `{hidden['adapter_dtype']}`; both finite: {hidden['bridge_finite']}/{hidden['adapter_finite']}. Attention mask exact: {hidden['attention_mask_exact']}.", "",
                  f"Max absolute error **{hidden['max_abs_error']:.9g}**; mean absolute error **{hidden['mean_abs_error']:.9g}**; RMSE **{hidden['rmse']:.9g}**; `np.allclose={hidden['allclose']}` with `rtol={hidden['rtol']}`, `atol={hidden['atol']}`.", "",
                  "## End-to-end frozen M0 prediction equivalence", "",
                  "PATH A uses the frozen Bridge clean text cache. PATH B uses this adapter's raw-text representation. Both use identical A2-valid audio/vision, P/O and one unchanged checkpoint per seed.", "",
                  "| Seed | Max abs logit Δ | Mean abs logit Δ | Class disagreements | Max abs intensity Δ | Mean abs intensity Δ |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for seed in SEEDS:
            p = result["prediction_equivalence"][str(seed)]
            lines.append(f"| {seed} | {p['max_abs_logit_difference']:.9g} | {p['mean_abs_logit_difference']:.9g} | {p['classification_disagreement_count']} | {p['max_abs_regression_difference']:.9g} | {p['mean_abs_regression_difference']:.9g} |")
        lines += ["", "| Seed | Path | Accuracy | macro-F1 | weighted-F1 | MAE | Pearson |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for seed in SEEDS:
            p = result["prediction_equivalence"][str(seed)]
            for path, key in (("A cache", "metrics_path_A"), ("B raw text", "metrics_path_B")):
                m = p[key]
                lines.append(f"| {seed} | {path} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['weighted_f1']:.6f} | {m['mae']:.6f} | {m['pearson']:.6f} |")
        lines += ["", "## Mask and future runner contract", "",
                  "The regenerated attention mask is exactly the A2 `text_P` and the tokenizer-padding `text_O`. M0 pools only positions with `P!=0`; padding hidden states cannot enter its final pooled prediction. The adapter makes no observation decision from hidden-state magnitude and adds no M1 logic.", "",
                  "A future unaligned A3 runner takes an **external official file/order key**, raw text, audio `[500,74]`, vision `[500,35]` and audited audio/vision P/O. It uses this adapter to create text hidden states and `text_P`, then sends text/audio/vision with masks to the frozen M0, retaining the external key for output correspondence. `NativeMulT.forward` does not require an internal sample `id`. Aligned A3 `text_bert` must not be mixed with unaligned A/V. This contract was defined without loading or predicting any A3 sample.", "",
                  "The validated claim is numerical equivalence of the A2 raw-text route and Bridge route. It does not validate future A3 sample content, missingness, performance or file-order execution.", ""]
    (OUT / "a3_text_adapter_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    try:
        result, rows = validate()
    except Exception as exc:
        result, rows = {"status": "BRIDGE_A3_TEXT_ADAPTER_BLOCKED", "error": repr(exc),
                        "scope": "A2 train/valid only; no A3 access"}, []
    write_outputs(result, rows)
    print(json.dumps({"status": result["status"],
                      "token_counts": result.get("token_equivalence"),
                      "hidden_max_error": result.get("hidden_state_equivalence", {}).get("max_abs_error")},
                     ensure_ascii=False), flush=True)
    if result["status"] == "BRIDGE_A3_TEXT_ADAPTER_BLOCKED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
