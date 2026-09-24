"""Resume Q3 Stage 1 from completed local-candidate CSVs after Q2 evidence gate."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path("workspace/experiments").resolve()))
import q3_stage1_validate as q
from q2_unaligned_text_adapter import file_sha256


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def restore_local(grids, data):
    result = {}
    for modality in q.ru.MODS:
        path = q.OUT / f"q3_local_{modality}_evidence.csv"
        saved = read_csv(path)
        generated = [c for row in grids[modality] for c in row]
        if len(saved) != len(generated):
            raise RuntimeError(f"saved {modality} candidate count mismatch")
        dc, dr = [], []
        for rec, cand in zip(saved, generated):
            row = cand["row"]
            if (int(rec["row"]) != row or rec["sample_id"] != data["samples"][row]["id"] or
                    rec["modality"] != modality or rec["scale"] != cand["scale"] or
                    int(rec["start_index_inclusive"]) != cand["start"] or
                    int(rec["end_index_exclusive"]) != cand["end"] or
                    int(rec["selected_eligible_positions"]) != len(cand["indices"])):
                raise RuntimeError(f"saved {modality} candidates differ from frozen grid")
            cand["Delta_cls"] = float(rec["Delta_cls"])
            cand["Delta_reg"] = float(rec["Delta_reg"])
            dc.append(cand["Delta_cls"])
            dr.append(cand["Delta_reg"])
        if not np.isfinite(dc).all() or not np.isfinite(dr).all():
            raise RuntimeError(f"saved {modality} responses nonfinite")
        result[modality] = {"candidate_count": len(generated), "sha256": file_sha256(path),
                            "Delta_cls": q.describe(dc), "Delta_reg": q.describe(dr)}
    return result


def verify_donor(data, full_z, full_y):
    path = q.OUT / "q3_donor_permutation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["seed"] != q.SEED or manifest["sample_ids_in_recipient_order"] != [s["id"] for s in data["samples"]]:
        raise RuntimeError("donor manifest/source mismatch")
    for modality in q.ru.MODS:
        perm = np.array(manifest["donor_row_by_modality"][modality], dtype=int)
        if len(perm) != q.N or set(perm.tolist()) != set(range(q.N)) or np.any(perm == np.arange(q.N)):
            raise RuntimeError("invalid frozen donor derangement")
    rows = read_csv(q.OUT / "q3_modality_contribution.csv")
    if len(rows) != q.N:
        raise RuntimeError("incomplete saved donor responses")
    for row, rec in enumerate(rows):
        cls = int(full_z[row].argmax())
        if (int(rec["row"]) != row or rec["sample_id"] != data["samples"][row]["id"] or
                int(rec["predicted_class_index"]) != cls or
                abs(float(rec["full_predicted_class_logit"])-float(full_z[row, cls])) > 1e-5 or
                abs(float(rec["full_intensity"])-float(full_y[row])) > 1e-5):
            raise RuntimeError("saved modality response baseline mismatch")
        for modality in q.ru.MODS:
            if int(rec[f"donor_{modality}_row"]) != manifest["donor_row_by_modality"][modality][row]:
                raise RuntimeError("saved donor row mismatch")
            if not np.isfinite(float(rec[f"Delta_cls_{modality}"])) or not np.isfinite(float(rec[f"Delta_reg_{modality}"])):
                raise RuntimeError("saved donor response nonfinite")
    summary = {}
    for modality in (*q.ru.MODS, "audio_vision"):
        suffix = modality if modality != "audio_vision" else "audio_vision_joint"
        summary[modality] = {
            "Delta_cls": q.describe([float(r[f"Delta_cls_{suffix}"]) for r in rows]),
            "Delta_reg": q.describe([float(r[f"Delta_reg_{suffix}"]) for r in rows]),
        }
    return summary, file_sha256(path)


def main():
    if (q.OUT / "q3_faithfulness.json").exists() or (q.OUT / "q3_stage1_review.md").exists():
        raise RuntimeError("Q3 Stage 1 review already exists")
    gate = json.loads(Path("workspace/results/q2_evidence/factorial_summary.json").read_text(encoding="utf-8"))
    if gate["status"] != "Q2_EVIDENCE_COMPLETE" or gate["cell_rows"] != 216:
        raise RuntimeError("Q2 evidence gate is not complete")
    started = time.perf_counter()
    data, text, adapter, model, _ = q.preflight()
    saved = [json.loads(line) for line in (q.RUN / "predictions_valid.jsonl").read_text(encoding="utf-8").splitlines()]
    clean = [r for r in saved if r["condition"] == "clean"]
    if len(clean) != q.N or any(r["row"] != i or r["sample_id"] != data["samples"][i]["id"]
                                    for i, r in enumerate(clean)):
        raise RuntimeError("frozen valid clean prediction mismatch")
    full_z = np.array([r["logits"] for r in clean], dtype=np.float32)
    full_y = np.array([r["predicted_intensity"] for r in clean], dtype=np.float32)
    donor_summary, donor_sha = verify_donor(data, full_z, full_y)
    grids = q.candidate_grid(data, adapter)
    local = restore_local(grids, data)
    print("Q3_COMPLETED_CANDIDATE_FILES_RESTORED", flush=True)
    faithful = q.faithfulness(grids, data, text, adapter, model, full_z, full_y)
    stable = q.stability(grids, data, text, adapter, model, full_z, full_y)
    text_pass = any(faithful["text"][target].get("passes_operator_consistency", False)
                    for target in ("cls", "reg"))
    av_pass = any(faithful[modality][target].get("passes_operator_consistency", False)
                  for modality in ("audio", "vision") for target in ("cls", "reg"))
    status = "Q3_EXPLANATION_METHOD_ACCEPTED" if text_pass and av_pass else "Q3_EXPLANATION_METHOD_NOT_FAITHFUL"
    payload = {"status": status, "scope": "A2 unaligned valid 728 only",
               "seed": q.SEED, "checkpoint_sha256": file_sha256(q.RUN / "checkpoint.pt"),
               "method_sha256": file_sha256(q.OUT / "q3_explanation_method.md"),
               "donor_manifest_sha256": donor_sha,
               "modality_contribution_csv_sha256": file_sha256(q.OUT / "q3_modality_contribution.csv"),
               "full_prediction_frozen_bridge_equivalent": True,
               "donor_aggregate": donor_summary, "local_evidence": local,
               "faithfulness": faithful, "stability": stable,
               "acceptance": {"text_pass_at_least_one_target": text_pass,
                              "audio_or_vision_pass_at_least_one_target": av_pass},
               "resumed_from_completed_candidate_files_after_q2_gate": True,
               "q2_gate_sha256": file_sha256(Path("workspace/results/q2_evidence/factorial_summary.json")),
               "runtime_seconds_resume": time.perf_counter()-started,
               "runtime_seconds": time.perf_counter()-started,
               "not_causal": True, "no_special_or_test_access": True}
    (q.OUT / "q3_faithfulness.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    q.report(payload)
    (q.OUT / "q3_stage1_paused.md").write_text(
        "# Q3 Stage 1 resumed after Q2 gate\n\nThe earlier process was interrupted after donor and local-candidate outputs. "
        "Those files were verified and reused without rerunning completed candidate scores. "
        "Faithfulness and stability were completed after `Q2_EVIDENCE_COMPLETE`; "
        f"see [review](q3_stage1_review.md). Decision: `{status}`.\n", encoding="utf-8")
    print(json.dumps({"status": status, "text_pass": text_pass, "av_pass": av_pass,
                      "seconds_resume": payload["runtime_seconds_resume"]}), flush=True)


if __name__ == "__main__":
    main()
