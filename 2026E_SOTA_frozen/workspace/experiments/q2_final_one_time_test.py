"""One-time frozen M0 evaluation on the official A2 unaligned test split."""
from __future__ import annotations

import csv
import json
import sys
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_recall_fscore_support)

sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
from q2_bridge import NativeMulT
from q2_unaligned_text_adapter import FrozenUnalignedTextAdapter, file_sha256

OUT = Path("workspace/results/q2_final")
FREEZE = OUT / "final_pipeline_freeze.json"
TEST = Path("workspace/data/processed/cleaning_2026e_v2/unaligned/test")
LABELS = Path("workspace/data/raw/E题数据/附件2-数据集特征文件/label.xlsx")
RUN = Path("workspace/experiments/q2_bridge_runs/M0_seed2718")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
EXPECTED_N = 727


def verify_freeze():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_A2_TEST_ACCESS" or freeze["selected_seed"] != 2718 or freeze["ensemble"]:
        raise RuntimeError("final predictor freeze is absent or changed")
    hashes = freeze["frozen_hashes_sha256"]
    paths = {
        "checkpoint_M0_seed2718": RUN / "checkpoint.pt",
        "M0_source_q2_bridge_py": Path("q2_bridge.py"),
        "M0_config_seed2718": RUN / "config.json",
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
        "A2_only_text_adapter_validation": Path("workspace/results/q2_bridge/a3_text_adapter_audit.json")}
    for key, path in paths.items():
        if file_sha256(path) != hashes[key]:
            raise RuntimeError(f"frozen hash mismatch: {key}")
    run_metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if run_metrics["status"] != "COMPLETE" or run_metrics["checkpoint_sha256"] != hashes["checkpoint_M0_seed2718"]:
        raise RuntimeError("selected frozen M0 run mismatch")
    return freeze


def xlsx_rows(path):
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//x:t", ns))
                      for si in root.findall("x:si", ns)]
        sheet = next(n for n in z.namelist()
                     if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        root = ET.fromstring(z.read(sheet))
        rows = []
        for row in root.findall(".//x:sheetData/x:row", ns):
            entry = {}
            for cell in row.findall("x:c", ns):
                col = "".join(c for c in cell.attrib.get("r", "") if c.isalpha())
                value = cell.find("x:v", ns)
                if cell.attrib.get("t") == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//x:t", ns))
                elif value is None:
                    text = ""
                elif cell.attrib.get("t") == "s":
                    text = shared[int(value.text)]
                else:
                    text = value.text
                entry[col] = text
            rows.append(entry)
    if not rows:
        raise RuntimeError("empty official label workbook")
    header = rows[0]
    result = [{header.get(col, col): value for col, value in row.items()} for row in rows[1:]]
    if not {"video_id", "clip_id", "label", "annotation", "mode"}.issubset(set(header.values())):
        raise RuntimeError("official label workbook fields changed")
    return result


def load_test_inputs():
    samples = json.loads((TEST / "samples.json").read_text(encoding="utf-8"))
    if len(samples) != EXPECTED_N or len({s["id"] for s in samples}) != EXPECTED_N:
        raise RuntimeError("test row count or IDs differ from frozen official split")
    if not all(s["row"] == i and isinstance(s["raw_text"], str) for i, s in enumerate(samples)):
        raise RuntimeError("test metadata row/raw_text mismatch")
    data = {"samples": samples}
    for name in ("audio", "vision", "audio_P", "audio_O", "vision_P", "vision_O",
                 "text_P", "text_O", "text_bert"):
        data[name] = np.load(TEST / (name + ".npy"), allow_pickle=False)
    if data["audio"].shape != (EXPECTED_N, 500, 74) or data["vision"].shape != (EXPECTED_N, 500, 35):
        raise RuntimeError("test native A/V shape mismatch")
    if data["text_bert"].shape != (EXPECTED_N, 3, 50):
        raise RuntimeError("test text triplet shape mismatch")
    for name in ("audio", "vision"):
        if not np.isfinite(data[name]).all():
            raise RuntimeError(f"nonfinite test {name}")
    for name, length in (("audio", 500), ("vision", 500), ("text", 50)):
        for state in ("P", "O"):
            array = data[f"{name}_{state}"]
            if array.shape != (EXPECTED_N, length) or not np.isin(array, (0, 1, 2)).all():
                raise RuntimeError(f"test {name}_{state} invalid")
    return data


def official_test_labels(samples):
    rows = xlsx_rows(LABELS)
    test_rows = [r for r in rows if r["mode"] == "test"]
    if len(test_rows) != EXPECTED_N:
        raise RuntimeError("official spreadsheet test count mismatch")
    indexed = {}
    for row in test_rows:
        sample_id = f"{row['video_id']}$_${row['clip_id']}"
        if sample_id in indexed:
            raise RuntimeError("duplicate official test label ID")
        indexed[sample_id] = row
    ids = [s["id"] for s in samples]
    if set(ids) != set(indexed):
        raise RuntimeError("processed test IDs and official spreadsheet differ")
    y = np.array([float(indexed[sample_id]["label"]) for sample_id in ids])
    if not np.isfinite(y).all():
        raise RuntimeError("nonfinite official test label")
    cls = (np.sign(y).astype(np.int64) + 1)
    if not np.array_equal(cls, np.array([CLASS_NAMES.index(indexed[sample_id]["annotation"])
                                        for sample_id in ids])):
        raise RuntimeError("official class annotation disagrees with frozen y sign mapping")
    return cls, y


def evaluate_metrics(true_cls, true_y, logits, intensity):
    predicted = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_cls, predicted, labels=[0, 1, 2], zero_division=0)
    pearson = (float(np.corrcoef(true_y, intensity)[0, 1])
               if np.std(true_y) and np.std(intensity) else None)
    return {"classification": {
                "accuracy": float(accuracy_score(true_cls, predicted)),
                "macro_f1": float(f1_score(true_cls, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(true_cls, predicted, labels=[0, 1, 2], average="weighted", zero_division=0)),
                "confusion_matrix_true_rows_predicted_columns": confusion_matrix(true_cls, predicted, labels=[0, 1, 2]).tolist(),
                "per_class": {CLASS_NAMES[i]: {"class_index": i, "precision": float(precision[i]),
                                              "recall": float(recall[i]), "f1": float(f1[i]),
                                              "support": int(support[i])} for i in range(3)}},
            "regression": {"mae": float(mean_absolute_error(true_y, intensity)),
                           "pearson": pearson}}


def write_success(result, records):
    with (OUT / "a2_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        columns = list(records[0])
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    result["prediction_csv_sha256"] = file_sha256(OUT / "a2_test_predictions.csv")
    (OUT / "a2_test_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    c, r = result["metrics"]["classification"], result["metrics"]["regression"]
    cm = c["confusion_matrix_true_rows_predicted_columns"]
    lines = ["# Q2 one-time official A2 test evaluation", "",
             "The final predictor was frozen in [final_pipeline_freeze.md](final_pipeline_freeze.md) before A2 test access: one M0 seed2718 checkpoint, raw-text adapter, no ensemble or calibration. This report is the one-time official **A2 unaligned test** evaluation. No model was trained, changed, reselected or tuned after seeing test results.", "",
             f"Test samples: **{result['test_samples']}**. Finite logits/intensity: **{result['finite_predictions']}**. Accuracy **{c['accuracy']:.4f}**, macro-F1 **{c['macro_f1']:.4f}**, weighted-F1 **{c['weighted_f1']:.4f}**; MAE **{r['mae']:.4f}**, Pearson **{r['pearson']:.4f}**.", "",
             "## Classification", "",
             "Confusion matrix, true rows and predicted columns in Negative / Neutral / Positive order:", "",
             "| True \\ Pred | Negative | Neutral | Positive |",
             "|---|---:|---:|---:|"]
    for name, row in zip(CLASS_NAMES, cm):
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", "| Class | Precision | Recall | F1 | Support |",
              "|---|---:|---:|---:|---:|"]
    for name in CLASS_NAMES:
        item = c["per_class"][name]
        lines.append(f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    lines += ["", "## Output and resources", "",
              f"Predicted class counts: {result['predicted_class_distribution']}; true class counts: {result['true_class_distribution']}.",
              f"Predicted intensity range: {result['output_range']['intensity']}; logit range: {result['output_range']['logits']}. BERT encoding {result['runtime_seconds']['bert_encoding']:.2f}s; M0 inference {result['runtime_seconds']['model_inference']:.2f}s; total {result['runtime_seconds']['total']:.2f}s. Sampled peak process RSS {result['peak_process_rss_gib']:.2f} GiB.", "",
              "## Evidence boundary", "",
              "[A2 valid S1–S6](../q2_bridge/M0_results.md) was development/model-selection and synthetic-gap robustness evidence. This A2 test report measures one-time clean generalization only; no test perturbations were generated. Attachment 3 remains reserved for future unlabeled inference. Differences from valid performance do not change the frozen predictor.", "",
              f"Official test labels came from Attachment 2 `label.xlsx` (SHA-256 `{result['label_workbook_sha256']}`), joined by the preserved official sample ID. Prediction table: [a2_test_predictions.csv](a2_test_predictions.csv) (SHA-256 `{result['prediction_csv_sha256']}`).", ""]
    (OUT / "a2_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def monitor_rss(stop, readings):
    proc = psutil.Process()
    while not stop.is_set():
        readings.append(proc.memory_info().rss)
        stop.wait(.05)
    readings.append(proc.memory_info().rss)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze()
    for path in (OUT / "test_execution_started.json", OUT / "a2_test_metrics.json",
                 OUT / "a2_test_predictions.csv", OUT / "a2_test_report.md"):
        if path.exists():
            raise RuntimeError("one-time A2 test evaluation was already started or completed")
    started = {"status": "ONE_TIME_TEST_EXECUTION_STARTED",
               "freeze_sha256": file_sha256(FREEZE),
               "selected_seed": 2718}
    with (OUT / "test_execution_started.json").open("x", encoding="utf-8") as f:
        json.dump(started, f, indent=2)
    t0 = time.perf_counter()
    stop = threading.Event()
    readings = []
    watcher = threading.Thread(target=monitor_rss, args=(stop, readings), daemon=True)
    watcher.start()
    try:
        data = load_test_inputs()
        adapter = FrozenUnalignedTextAdapter()
        bt = time.perf_counter()
        text = adapter.encode_raw_text([s["raw_text"] for s in data["samples"]])
        bert_seconds = time.perf_counter() - bt
        if not np.array_equal(text["text_bert"], data["text_bert"]):
            raise RuntimeError("test raw_text tokenization differs from official A2 text_bert")
        if not np.array_equal(text["text_P"], data["text_P"]) or not np.array_equal(text["text_O"], data["text_O"]):
            raise RuntimeError("test text P/O differs from frozen tokenizer padding")
        data["text_P"] = text["text_P"]
        data["text_O"] = text["text_O"]
        model = NativeMulT(False)
        model.load_state_dict(torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=True))
        model.eval()
        mt = time.perf_counter()
        logits, intensity = ru.predict_all(model, data, text["text_hidden"], None, {}, None)
        model_seconds = time.perf_counter() - mt
        finite = bool(np.isfinite(logits).all() and np.isfinite(intensity).all())
        if logits.shape != (EXPECTED_N, 3) or intensity.shape != (EXPECTED_N,) or not finite:
            raise RuntimeError("nonfinite or wrong-shape test prediction")
        # Labels are first opened after the final predictions are computed.
        true_cls, true_y = official_test_labels(data["samples"])
        metrics = evaluate_metrics(true_cls, true_y, logits, intensity)
        predicted = logits.argmax(axis=1)
        records = [{"row": i, "sample_id": data["samples"][i]["id"],
                    "logit_negative": float(logits[i, 0]),
                    "logit_neutral": float(logits[i, 1]),
                    "logit_positive": float(logits[i, 2]),
                    "predicted_class": int(predicted[i]),
                    "predicted_polarity": CLASS_NAMES[int(predicted[i])],
                    "predicted_intensity": float(intensity[i]),
                    "true_class": int(true_cls[i]),
                    "true_polarity": CLASS_NAMES[int(true_cls[i])],
                    "true_intensity": float(true_y[i])} for i in range(EXPECTED_N)]
        stop.set()
        watcher.join()
        result = {"status": "Q2_FINAL_PIPELINE_FROZEN_AND_TESTED",
                  "test_samples": EXPECTED_N,
                  "selected_seed": freeze["selected_seed"],
                  "predictor": "single frozen M0 seed2718",
                  "freeze_sha256": file_sha256(FREEZE),
                  "checkpoint_sha256": file_sha256(RUN / "checkpoint.pt"),
                  "label_workbook_sha256": file_sha256(LABELS),
                  "finite_predictions": finite,
                  "output_range": {"logits": [float(logits.min()), float(logits.max())],
                                   "intensity": [float(intensity.min()), float(intensity.max())]},
                  "predicted_class_distribution": {name: int(np.sum(predicted == i))
                                                   for i, name in enumerate(CLASS_NAMES)},
                  "true_class_distribution": {name: int(np.sum(true_cls == i))
                                              for i, name in enumerate(CLASS_NAMES)},
                  "runtime_seconds": {"bert_encoding": bert_seconds,
                                      "model_inference": model_seconds,
                                      "total": time.perf_counter() - t0},
                  "peak_process_rss_bytes": max(readings),
                  "peak_process_rss_gib": max(readings) / 1024 ** 3,
                  "text_triplet_exact_to_frozen_A2_test": True,
                  "text_P_O_exact_to_frozen_A2_test": True,
                  "metrics": metrics,
                  "development_robustness_source": "workspace/results/q2_bridge/M0_results.md (A2 valid S1-S6)",
                  "no_test_perturbations_or_reselection": True}
        write_success(result, records)
        print(json.dumps({"status": result["status"], "n": EXPECTED_N,
                          "macro_f1": metrics["classification"]["macro_f1"],
                          "mae": metrics["regression"]["mae"]}), flush=True)
    except Exception as exc:
        stop.set()
        watcher.join()
        failure = {"status": "Q2_TEST_EVALUATION_TECHNICAL_FAILURE",
                   "error": repr(exc), "freeze_sha256": file_sha256(FREEZE),
                   "selected_seed": 2718, "elapsed_seconds": time.perf_counter() - t0,
                   "peak_process_rss_bytes": max(readings) if readings else None}
        (OUT / "a2_test_metrics.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        (OUT / "a2_test_report.md").write_text(
            "# Q2 one-time A2 test evaluation\n\nTechnical failure after frozen execution started. "
            f"Error: `{failure['error']}`. No model reselection or tuning was performed.\n",
            encoding="utf-8")
        print(json.dumps(failure), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
