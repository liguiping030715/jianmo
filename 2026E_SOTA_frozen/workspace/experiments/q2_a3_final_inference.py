"""One-time frozen Q2 inference on Attachment 3, unaligned version only."""
from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
from q2_bridge import NativeMulT
from q2_unaligned_text_adapter import FrozenUnalignedTextAdapter, file_sha256

OUT = Path("workspace/results/q2_final")
FREEZE = OUT / "final_pipeline_freeze.json"
PROCESSED = Path("workspace/data/processed/cleaning_2026e_v2/unaligned/attachment3")
RAW = Path("workspace/data/raw")
CHECKPOINT = Path("workspace/experiments/q2_bridge_runs/M0_seed2718/checkpoint.pt")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
N = 30


def preflight():
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (frozen["status"] != "FROZEN_BEFORE_A2_TEST_ACCESS" or
            frozen["selected_seed"] != 2718 or frozen["ensemble"]):
        raise RuntimeError("final M0 seed2718 predictor is not frozen as expected")
    paths = {
        "checkpoint_M0_seed2718": CHECKPOINT,
        "M0_source_q2_bridge_py": Path("q2_bridge.py"),
        "M0_config_seed2718": CHECKPOINT.parent / "config.json",
        "text_adapter_source": Path("q2_unaligned_text_adapter.py"),
        "tokenizer_vocab": Path("workspace/references/bert-base-uncased/vocab.txt"),
        "tokenizer_json": Path("workspace/references/bert-base-uncased/tokenizer.json"),
        "tokenizer_config": Path("workspace/references/bert-base-uncased/tokenizer_config.json"),
        "BERT_weights": Path("workspace/references/bert-base-uncased/pytorch_model.bin"),
        "BERT_config": Path("workspace/references/bert-base-uncased/config.json"),
        "cleaning_v2_manifest": Path("workspace/data/processed/cleaning_2026e_v2/manifest.json"),
        "unaligned_train_fitted_scaler": Path("workspace/data/processed/cleaning_2026e_v2/unaligned/scalers.json"),
        "cleaning_contract": Path("workspace/evidence_recovery/cleaning_contract.md"),
        "observation_contract": Path("workspace/evidence_recovery/observation_contract.md"),
        "A2_only_text_adapter_validation": Path("workspace/results/q2_bridge/a3_text_adapter_audit.json"),
    }
    for name, path in paths.items():
        if file_sha256(path) != frozen["frozen_hashes_sha256"][name]:
            raise RuntimeError(f"frozen checksum mismatch: {name}")
    manifest = json.loads(paths["cleaning_v2_manifest"].read_text(encoding="utf-8"))
    return frozen, manifest


def load_unaligned_a3(manifest):
    samples = json.loads((PROCESSED / "samples.json").read_text(encoding="utf-8"))
    if len(samples) != N:
        raise RuntimeError(f"expected {N} official A3 unaligned files")
    source_files = []
    local_keys = []
    for i, sample in enumerate(samples):
        source = sample["source_file"]
        key = sample["local_key"]
        expected_key = f"附件3_未对齐版本_{i + 1:02d}"
        if (sample["row"] != i or sample.get("id") is not None or
                sample.get("original_id_present") is not False or
                key != expected_key or Path(source).stem != key or
                not re.fullmatch(r"附件3_未对齐版本_\d{2}\.pkl", Path(source).name) or
                not isinstance(sample.get("raw_text"), str) or not sample["raw_text"].strip()):
            raise RuntimeError(f"official file/order/raw_text mismatch at row {i}")
        raw_file = RAW / source
        if not raw_file.is_file() or manifest["source_sha256"].get(source) != file_sha256(raw_file):
            raise RuntimeError(f"official source-file hash mismatch: {source}")
        source_files.append(source)
        local_keys.append(key)
    if len(set(source_files)) != N or len(set(local_keys)) != N:
        raise RuntimeError("official A3 external file/order keys are not unique")
    raw_folder = (RAW / source_files[0]).parent
    if {p.name for p in raw_folder.glob("*.pkl")} != {Path(s).name for s in source_files}:
        raise RuntimeError("official A3 unaligned folder does not match the 30 frozen keys")

    data = {"samples": samples}
    for name in ("audio", "vision"):
        data[name] = np.load(PROCESSED / f"{name}.npy", allow_pickle=False)
        data[f"{name}_P"] = np.load(PROCESSED / f"{name}_P.npy", allow_pickle=False)
        data[f"{name}_O"] = np.load(PROCESSED / f"{name}_O.npy", allow_pickle=False)
        data[f"{name}_zero_context"] = np.load(PROCESSED / f"{name}_zero_context.npy", allow_pickle=False)
        expected_shape = (N, 500, 74 if name == "audio" else 35)
        values = data[name]
        if values.shape != expected_shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid frozen {name} tensor")
        for state in ("P", "O"):
            a = data[f"{name}_{state}"]
            if a.shape != (N, 500) or a.dtype != np.uint8 or not np.isin(a, (0, 1, 2)).all():
                raise RuntimeError(f"invalid frozen {name}_{state}")
        context = data[f"{name}_zero_context"]
        if context.shape != (N, 500) or context.dtype != np.uint8 or not np.isin(context, (0, 2, 3)).all():
            raise RuntimeError(f"invalid frozen {name}_zero_context")
    return data


def descriptive_runs(data):
    rows = []
    summary = {}
    for modality in ("audio", "vision"):
        arr = data[modality]
        zero = np.all(arr == 0, axis=-1)
        context = data[f"{modality}_zero_context"]
        run_kinds = Counter()
        lengths = []
        sample_ratios = []
        for i, sample in enumerate(data["samples"]):
            sample_ratio = float(np.mean(zero[i]))
            sample_ratios.append(sample_ratio)
            boundaries = np.flatnonzero(np.diff(np.r_[False, zero[i], False].astype(np.int8)))
            for start, end in zip(boundaries[::2], boundaries[1::2]):
                start, end = int(start), int(end)
                if start == 0 and end == 500:
                    kind = "whole_sequence_zero_ambiguous"
                elif start == 0:
                    kind = "prefix_zero_ambiguous"
                elif end == 500:
                    kind = "suffix_zero_ambiguous"
                else:
                    kind = "interior_zero_candidate"
                length = end - start
                run_kinds[kind] += 1
                lengths.append(length)
                rows.append({
                    "order_1based": i + 1,
                    "external_file_order_key": sample["local_key"],
                    "official_source_file": sample["source_file"],
                    "modality": modality,
                    "zero_run_type": kind,
                    "start_index_inclusive": start,
                    "end_index_exclusive": end,
                    "length_native_steps": length,
                    "run_ratio_of_500": length / 500,
                    "relative_start": start / 500,
                    "relative_end": end / 500,
                    "sample_zero_row_ratio": sample_ratio,
                    "frozen_P_values": ";".join(map(str, sorted(set(data[f"{modality}_P"][i, start:end].tolist())))),
                    "frozen_O_values": ";".join(map(str, sorted(set(data[f"{modality}_O"][i, start:end].tolist())))),
                    "zero_context_3_positions_diagnostic_only": int(np.sum(context[i, start:end] == 3)),
                    "certified_missingness": False,
                })
        summary[modality] = {
            "zero_rows": int(zero.sum()),
            "total_rows": int(zero.size),
            "zero_row_ratio": float(zero.mean()),
            "nonzero_rows": int((~zero).sum()),
            "run_count": len(lengths),
            "run_types": dict(run_kinds),
            "run_length_native_steps": {"min": min(lengths) if lengths else None,
                                        "median": float(np.median(lengths)) if lengths else None,
                                        "max": max(lengths) if lengths else None},
            "sample_zero_row_ratio": {"min": min(sample_ratios),
                                      "median": float(np.median(sample_ratios)),
                                      "max": max(sample_ratios)},
            "zero_context_3_positions_diagnostic_only": int(np.sum(context == 3)),
            "frozen_P_counts": {str(v): int(n) for v, n in zip(*np.unique(data[f"{modality}_P"], return_counts=True))},
            "frozen_O_counts": {str(v): int(n) for v, n in zip(*np.unique(data[f"{modality}_O"], return_counts=True))},
        }
    return rows, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def monitor_rss(stop, readings):
    proc = psutil.Process()
    while not stop.is_set():
        readings.append(proc.memory_info().rss)
        stop.wait(.05)
    readings.append(proc.memory_info().rss)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen, manifest = preflight()
    outputs = [OUT / name for name in ("a3_final_predictions.csv", "a3_final_inference_report.md",
                                      "a3_final_inference.json", "a3_missingness_descriptive.csv")]
    lock = OUT / "a3_final_execution_started.json"
    if lock.exists() or any(path.exists() for path in outputs):
        raise RuntimeError("one-time A3 final inference was already started or completed")
    lock.write_text(json.dumps({"status": "ONE_TIME_A3_INFERENCE_STARTED",
                                "freeze_sha256": file_sha256(FREEZE),
                                "checkpoint_sha256": file_sha256(CHECKPOINT)}, indent=2), encoding="utf-8")
    started = time.perf_counter()
    stop = threading.Event()
    readings = []
    watcher = threading.Thread(target=monitor_rss, args=(stop, readings), daemon=True)
    watcher.start()
    try:
        data = load_unaligned_a3(manifest)
        adapter = FrozenUnalignedTextAdapter()
        bert_start = time.perf_counter()
        text = adapter.encode_raw_text([s["raw_text"] for s in data["samples"]])
        bert_seconds = time.perf_counter() - bert_start
        if (text["text_hidden"].shape != (N, 50, 768) or text["text_hidden"].dtype != np.float32 or
                not np.isfinite(text["text_hidden"]).all() or text["text_bert"].shape != (N, 3, 50) or
                not np.array_equal(text["text_P"], text["text_bert"][:, 1, :]) or
                not np.array_equal(text["text_O"], text["text_P"]) or
                np.any(text["text_P"].sum(axis=1) < 2)):
            raise RuntimeError("A3 regenerated BERT text interface or attention mask is invalid")
        data["text_P"], data["text_O"] = text["text_P"], text["text_O"]
        model = NativeMulT(False)
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        model.eval()
        model_start = time.perf_counter()
        logits, intensity = ru.predict_all(model, data, text["text_hidden"], None, {}, None)
        model_seconds = time.perf_counter() - model_start
        if logits.shape != (N, 3) or intensity.shape != (N,) or not (np.isfinite(logits).all() and np.isfinite(intensity).all()):
            raise RuntimeError("A3 predictions are nonfinite or have wrong shape")
        predicted = logits.argmax(axis=1)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        predictions = [{
            "order_1based": i + 1,
            "external_file_order_key": sample["local_key"],
            "official_source_file": sample["source_file"],
            "predicted_class_index": int(predicted[i]),
            "predicted_polarity": CLASS_NAMES[int(predicted[i])],
            "predicted_intensity": float(intensity[i]),
            "logit_negative": float(logits[i, 0]),
            "logit_neutral": float(logits[i, 1]),
            "logit_positive": float(logits[i, 2]),
            "prob_negative": float(probs[i, 0]),
            "prob_neutral": float(probs[i, 1]),
            "prob_positive": float(probs[i, 2]),
        } for i, sample in enumerate(data["samples"])]
        runs, missingness = descriptive_runs(data)
        if not runs:
            raise RuntimeError("no zero-run rows for requested descriptive report")
        stop.set()
        watcher.join()
        write_csv(outputs[0], predictions)
        write_csv(outputs[3], runs)
        result = {
            "status": "Q2_ATTACHMENT3_FINAL_INFERENCE_COMPLETE",
            "scope": "one-time unlabeled Attachment 3 unaligned inference; no model selection",
            "sample_count": N,
            "official_order": "frozen samples.json row 0..29 maps exactly to official unaligned files 01..30",
            "internal_id_required": False,
            "predictor": "single frozen M0 seed2718",
            "freeze_sha256": file_sha256(FREEZE),
            "checkpoint_sha256": file_sha256(CHECKPOINT),
            "processed_A3_input_sha256": {p.name: file_sha256(p) for p in sorted(PROCESSED.iterdir()) if p.is_file()},
            "text_adapter": {"source_sha256": file_sha256(Path("q2_unaligned_text_adapter.py")),
                             "provenance": adapter.provenance(),
                             "text_hidden_shape": list(text["text_hidden"].shape),
                             "text_hidden_dtype": str(text["text_hidden"].dtype),
                             "attention_mask_valid_token_count": int(text["text_P"].sum()),
                             "attention_mask_padding_token_count": int(text["text_P"].size - text["text_P"].sum()),
                             "attention_mask_integrity": True},
            "input_audit": {m: {"shape": list(data[m].shape), "dtype": str(data[m].dtype),
                                "finite": True, "P_values": missingness[m]["frozen_P_counts"],
                                "O_values": missingness[m]["frozen_O_counts"]}
                            for m in ("audio", "vision")},
            "prediction_audit": {"finite": True,
                                 "predicted_class_counts": {name: int(np.sum(predicted == j)) for j, name in enumerate(CLASS_NAMES)},
                                 "intensity_range": [float(intensity.min()), float(intensity.max())],
                                 "logit_range": [float(logits.min()), float(logits.max())],
                                 "probability_row_sum_max_abs_error": float(np.max(np.abs(probs.sum(axis=1) - 1)))},
            "descriptive_zero_runs": missingness,
            "missingness_interpretation": "zero runs are observable candidates only, not certified missing masks; index positions, not physical seconds",
            "runtime_seconds": {"bert_encoding": bert_seconds, "model_inference": model_seconds,
                                "total": time.perf_counter() - started},
            "peak_process_rss_bytes": max(readings),
            "predictions_csv_sha256": file_sha256(outputs[0]),
            "descriptive_csv_sha256": file_sha256(outputs[3]),
            "no_training_tuning_imputation_or_reselection": True,
        }
        outputs[2].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        lines = ["# Q2 Attachment 3 final inference", "",
                 "The permanently frozen **M0 seed2718** model ran once on the 30 official Attachment 3 unaligned files in order 01–30. The external file/order key is preserved; no internal sample ID was required. No training, tuning, threshold change, ensemble, imputation, smoothing, or sample removal occurred.", "",
                 "## Frozen input interface", "",
                 "`raw_text` → validated pinned tokenizer → frozen BERT → 50×768; native unaligned audio 500×74 and vision 500×35 came from frozen `cleaning_2026e_v2`. Aligned Attachment 3 text was not used. Text attention mask became text P/O. Audio/vision P/O were read unchanged from the frozen arrays; M0 excludes P=0 only and does not infer missingness from feature magnitude.", "",
                 f"All 30 texts were available; BERT produced {result['text_adapter']['attention_mask_valid_token_count']} valid and {result['text_adapter']['attention_mask_padding_token_count']} padded token positions. Inputs and predictions were finite. Predicted class counts: {result['prediction_audit']['predicted_class_counts']}. Intensity range: {result['prediction_audit']['intensity_range']}.", "",
                 "## Observable zero intervals, separate from prediction", "",
                 "Counts below describe exact all-zero feature rows in the frozen input. These are **not certified missing labels**. Interior runs are candidates; prefix, suffix, and whole-sequence runs have additional padding/content ambiguity. `zero_context=3` is diagnostic only. Duration is in native index steps and relative fraction of 500; no verified index-to-seconds map is asserted.", "",
                 "| Modality | Zero rows / 15,000 | Ratio | Runs | Interior | Prefix | Suffix | Whole | Median run steps |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for modality in ("audio", "vision"):
            item = missingness[modality]
            types = item["run_types"]
            lines.append(f"| {modality} | {item['zero_rows']} | {item['zero_row_ratio']:.4f} | {item['run_count']} | {types.get('interior_zero_candidate', 0)} | {types.get('prefix_zero_ambiguous', 0)} | {types.get('suffix_zero_ambiguous', 0)} | {types.get('whole_sequence_zero_ambiguous', 0)} | {item['run_length_native_steps']['median']:.1f} |")
        lines += ["", "## Artifacts and provenance", "",
                  "- [a3_final_predictions.csv](a3_final_predictions.csv): official file order, class index/polarity, intensity, raw logits and derived probabilities.",
                  "- [a3_missingness_descriptive.csv](a3_missingness_descriptive.csv): every observable zero interval with native and relative positions; no fabricated missing mask.",
                  "- [a3_final_inference.json](a3_final_inference.json): input checks, frozen hashes, adapter settings, runtime and full descriptive summary.", "",
                  f"BERT encoding {bert_seconds:.2f}s; M0 inference {model_seconds:.2f}s; total {result['runtime_seconds']['total']:.2f}s; sampled peak RSS {result['peak_process_rss_bytes'] / 1024**3:.2f} GiB.", "",
                  "Attachment 3 has no evaluation labels in this path. These outputs are predictions and descriptive input statistics, not measured accuracy or certified missingness.", ""]
        outputs[1].write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"status": result["status"], "samples": N,
                          "predicted_class_counts": result["prediction_audit"]["predicted_class_counts"]}), flush=True)
    except Exception as exc:
        stop.set()
        watcher.join()
        failure = {"status": "Q2_ATTACHMENT3_TECHNICAL_FAILURE", "error": repr(exc),
                   "freeze_sha256": file_sha256(FREEZE), "checkpoint_sha256": file_sha256(CHECKPOINT),
                   "elapsed_seconds": time.perf_counter() - started}
        outputs[2].write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs[1].write_text("# Q2 Attachment 3 final inference\n\nOne-time execution encountered a technical failure: `" + repr(exc) + "`.\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
