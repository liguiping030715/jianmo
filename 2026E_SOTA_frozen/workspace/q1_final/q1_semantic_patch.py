"""Q1 semantic repair; no media decoding or temporal alignment is performed.

The original extractor discarded word-level BERT outputs after 50-bin projection.
This script reruns only the frozen BERT on the official transcripts to preserve
sample-level content for all 100 clips. It is idempotent after the first run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEAT = HERE / "features"
PROC = ROOT / "workspace" / "data" / "processed" / "cleaning_2026e_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import torch
    from transformers import BertModel, BertTokenizerFast

    torch.set_num_threads(4)
    freeze = json.loads((HERE / "q1_extractor_freeze.json").read_text(encoding="utf-8"))
    baseline = json.loads((HERE / "q1_semantic_patch_source_hashes.json").read_text(encoding="utf-8"))
    for name, expected in baseline.items():
        assert sha(ROOT / name) == expected, f"frozen source/model changed: {name}"
    bert_dir = Path(freeze["text_embedding"]["local_path"])
    assert sha(bert_dir / "pytorch_model.bin") == freeze["text_embedding"]["weights_sha256"]
    assert sha(bert_dir / "vocab.txt") == freeze["text_embedding"]["vocab_sha256"]

    official = {r["id"]: r for r in json.loads((PROC / "attachment1_manifest.json").read_text(encoding="utf-8"))}
    manifest = csv_rows(HERE / "q1_manifest.csv")
    alignment = csv_rows(HERE / "q1_alignment_report.csv")
    assert len(official) == len(manifest) == len(alignment) == 100
    assert {r["id"] for r in manifest} == set(official) == {r["id"] for r in alignment}
    assert sum(r["text_alignment_status"] == "VERIFIED" for r in manifest) == 78
    assert sum(r["text_alignment_status"] == "UNRESOLVED" for r in manifest) == 22

    # Lazy loading: a second invocation audits existing content without BERT work.
    def lacks_words(r: dict) -> bool:
        with np.load(FEAT / f'{r["id"]}.npz', allow_pickle=False) as archive:
            return "text_word_feat" not in archive.files

    need_bert = any(lacks_words(r) for r in manifest)
    if need_bert:
        tok = BertTokenizerFast.from_pretrained(str(bert_dir), local_files_only=True)
        model = BertModel.from_pretrained(str(bert_dir), local_files_only=True).eval()
    else:
        tok = model = None

    align_by_id = {r["id"]: r for r in alignment}
    provenance = json.loads((HERE / "all_provenance.json").read_text(encoding="utf-8"))
    prov_by_id = {r["id"]: r for r in provenance}
    assert set(prov_by_id) == set(official)
    for i, row in enumerate(manifest):
        rid = row["id"]
        rec = official[rid]
        p = FEAT / f"{rid}.npz"
        jpath = FEAT / f"{rid}.json"
        prov = json.loads(jpath.read_text(encoding="utf-8"))
        assert prov["text"] == rec["text"] and prov["text"] == prov_by_id[rid]["text"]
        assert prov["label"] == rec["label"] and prov["annotation"] == rec["annotation"]
        status = prov["text_alignment"]["status"]
        assert status == row["text_alignment_status"]
        state = 1 if status == "VERIFIED" else 2
        with np.load(p, allow_pickle=False) as old:
            data = {key: old[key] for key in old.files}
        if "text_word_feat" not in data:
            enc = tok(rec["text"], return_offsets_mapping=True, truncation=True,
                      max_length=512, return_tensors="pt")
            with torch.inference_mode():
                hidden = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               token_type_ids=enc["token_type_ids"]).last_hidden_state[0].numpy()
            word_ids = enc.word_ids(0)
            spans = enc["offset_mapping"][0].numpy()
            groups: dict[int, list[int]] = {}
            for sub_idx, wid in enumerate(word_ids):
                if wid is not None:
                    groups.setdefault(wid, []).append(sub_idx)
            starts, ends, vectors, surface = [], [], [], []
            for wid in sorted(groups):
                idxs = groups[wid]
                cs, ce = int(spans[idxs, 0].min()), int(spans[idxs, 1].max())
                starts.append(cs)
                ends.append(ce)
                vectors.append(hidden[idxs].mean(0).astype(np.float32))
                surface.append(rec["text"][cs:ce])
            assert vectors, f"no BERT words: {rid}"
            data["text_word_feat"] = np.stack(vectors)
            data["text_word_char_start"] = np.asarray(starts, dtype=np.int32)
            data["text_word_char_end"] = np.asarray(ends, dtype=np.int32)
            data["text_word_surface"] = np.asarray(surface, dtype=np.str_)
            data["text_utterance_feat"] = data["text_word_feat"].mean(0).astype(np.float32)
        assert len(data["text_word_feat"]) == int(row["n_words"])
        assert np.isfinite(data["text_word_feat"]).all()
        assert np.isfinite(data["text_utterance_feat"]).all()

        # 2 means location unknown. No word times or 50-bin embedding are invented.
        if state == 2:
            assert not np.any(data["text_P"] == 1) and not np.any(data["text_O"] == 1)
            data["text_P"] = np.full(50, 2, dtype=np.uint8)
            data["text_O"] = np.full(50, 2, dtype=np.uint8)
        else:
            assert set(np.unique(data["text_P"])).issubset({0, 1, 2})
            assert set(np.unique(data["text_O"])).issubset({0, 1, 2})
            # A sample-level pass does not establish that every other bin is
            # speech-free: low-confidence words may still occupy those bins.
            data["text_P"][data["text_P"] == 0] = 2
            data["text_O"][data["text_O"] == 0] = 2
        data["text_alignment_state"] = np.asarray(state, dtype=np.uint8)
        data["text_content_present"] = np.asarray(1, dtype=np.uint8)
        # Explicit finite numerical inputs; zero here is a placeholder only.
        for mod in ("text", "audio", "vision"):
            f = data[f"{mod}_feat"]
            o = data[f"{mod}_O"]
            assert np.isfinite(f[o == 1]).all()
            safe = np.where(o[:, None] == 1, f, 0.0).astype(np.float32)
            assert np.isfinite(safe).all()
            data[f"{mod}_model_input"] = safe
        with (p.with_suffix(".npz.tmp")).open("wb") as f:
            np.savez_compressed(f, **data)
        os.replace(p.with_suffix(".npz.tmp"), p)

        for item in (prov, prov_by_id[rid]):
            item["text_alignment_state"] = state
            item["text_content_present"] = True
            item["text_content_representation"] = "text_word_feat plus text_utterance_feat"
            item["temporal_text_representation"] = "50-bin only where forced alignment VERIFIED"
            item["bin_P_unknown_counts"] = {"text": int(np.count_nonzero(data["text_P"] == 2))}
            item["bin_O_unknown_counts"] = {"text": int(np.count_nonzero(data["text_O"] == 2))}
        jpath.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
        row["text_alignment_state"] = state
        row["text_content_present"] = 1
        row["text_word_content_count"] = len(data["text_word_feat"])
        row["text_P_bins"] = int(np.count_nonzero(data["text_P"] == 1))
        row["text_O_bins"] = int(np.count_nonzero(data["text_O"] == 1))
        row["text_P_unknown_bins"] = int(np.count_nonzero(data["text_P"] == 2))
        row["text_O_unknown_bins"] = int(np.count_nonzero(data["text_O"] == 2))
        row["npz_sha256"] = sha(p)
        ar = align_by_id[rid]
        ar["text_alignment_status"] = status
        ar["text_alignment_state"] = state
        ar["text_content_present"] = 1
        ar["text_word_content_count"] = len(data["text_word_feat"])
        ar["text_P_unknown_bins"] = row["text_P_unknown_bins"]
        ar["text_O_unknown_bins"] = row["text_O_unknown_bins"]
        if i % 10 == 0:
            print(f"patched {i + 1}/100", flush=True)

    write_csv(HERE / "q1_manifest.csv", manifest)
    write_csv(HERE / "q1_alignment_report.csv", [align_by_id[r["id"]] for r in manifest])
    (HERE / "all_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print("semantic patch complete: VERIFIED=78, UNRESOLVED=22, content=100")


if __name__ == "__main__":
    main()
