"""Fixed-weight A2-valid shadow interface probe; no training or A3 access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

import sys
sys.path.insert(0, str(Path.cwd()))
import pilot_ru as ru
from q2_bridge import NativeMulT

BASE = Path("workspace/data/processed/cleaning_2026e_v2/unaligned/valid")
CACHE = Path("workspace/experiments/pilot_cache/clean_valid.npy")
RUNS = Path("workspace/experiments/q2_bridge_runs")
OUT = Path("workspace/results/q2_bridge")
BERT = Path("workspace/references/bert-base-uncased")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_forward(level, states, data, rows):
    model = NativeMulT(level == "M1")
    checkpoint = RUNS / f"{level}_seed1729/checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        parts = []
        for row in rows:
            parts.append((states[row], data["audio"][row], data["vision"][row],
                          [data[f"{m}_P"][row] for m in ru.MODS],
                          [data[f"{m}_O"][row] for m in ru.MODS]))
        batch = ru.stack_batch(parts)
        z, y, _ = model(*batch)
    return z.numpy(), y.numpy(), model, batch


def main():
    ru.set_determinism(1729)
    data = ru.load_split("valid")
    samples = data["samples"]
    assert len(samples) == 728
    tokenizer = BertTokenizerFast.from_pretrained(str(BERT), local_files_only=True)
    encoded = tokenizer([s["raw_text"] for s in samples], padding="max_length",
                        truncation=True, max_length=50, return_tensors="np")
    tokens = np.stack([encoded["input_ids"], encoded["attention_mask"],
                       encoded["token_type_ids"]], axis=1)
    if not np.array_equal(tokens, data["text_bert"]):
        raise RuntimeError("A2-valid raw_text tokenizer differs from supplied text_bert")
    if sha(BERT / "pytorch_model.bin") != "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a":
        raise RuntimeError("BERT changed")
    encoder = BertModel.from_pretrained(str(BERT), local_files_only=True).eval()
    state = np.empty((728, 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, 728, 16):
            t = torch.from_numpy(tokens[start:start+16].copy())
            state[start:start+len(t)] = encoder(input_ids=t[:, 0],
                                                 attention_mask=t[:, 1],
                                                 token_type_ids=t[:, 2]).last_hidden_state.numpy()
    cached = np.load(CACHE, mmap_mode="r")
    max_state_diff = float(np.max(np.abs(state - cached)))
    if max_state_diff > 1e-5:
        raise RuntimeError(f"recomputed states differ: {max_state_diff}")
    del encoder

    rows = list(range(32))
    probe = {"scope": "A2 valid only; seed1729 fixed M0/M1 checkpoints; no retraining/A3 access",
             "raw_text_rows": 728, "token_triplets_exact": 728,
             "bert_state_shape": [728, 50, 768], "max_abs_BERT_state_difference": max_state_diff,
             "padded_positions_in_probe_batch": int((data["text_P"][rows] == 0).sum()),
             "fixed_checkpoint_results": {}}
    for level in ("M0", "M1"):
        z_cache, y_cache, model, batch = model_forward(level, cached, data, rows)
        z_raw, y_raw, _, _ = model_forward(level, state, data, rows)
        logits_diff = float(np.max(np.abs(z_cache - z_raw)))
        intensity_diff = float(np.max(np.abs(y_cache - y_raw)))
        # Stress padded query rows only; this tests whether those BERT states
        # can influence the final prediction through cross-attention.
        altered = [x.clone() if torch.is_tensor(x) else [v.clone() for v in x]
                   for x in batch]
        pad = altered[3][0] == 0
        altered[0][pad] = 1e4
        with torch.inference_mode():
            z_pad, y_pad, _ = model(*altered)
        pad_logit_diff = float(np.max(np.abs(z_cache - z_pad.numpy())))
        pad_intensity_diff = float(np.max(np.abs(y_cache - y_pad.numpy())))
        if max(logits_diff, intensity_diff, pad_logit_diff, pad_intensity_diff) > 1e-5:
            raise RuntimeError(f"fixed-weight input/padding invariance failed: {level}")
        probe["fixed_checkpoint_results"][level] = {
            "checkpoint_sha256": sha(RUNS / f"{level}_seed1729/checkpoint.pt"),
            "rows": len(rows), "max_abs_logit_difference_raw_text_vs_cache": logits_diff,
            "max_abs_intensity_difference_raw_text_vs_cache": intensity_diff,
            "max_abs_logit_difference_padding_stress": pad_logit_diff,
            "max_abs_intensity_difference_padding_stress": pad_intensity_diff}
    probe["result"] = "A2_RAW_TEXT_SHADOW_PATH_EQUIVALENT"
    probe["A3_same_version_schema_limitation"] = "Unaligned A3 supplies raw_text, not a supplied text_bert field; no A3 sample/inference was used."
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interface_probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    lines = ["# Fixed-weight A2-valid text interface probe", "",
             "All 728 A2 valid `raw_text` strings reproduced the stored `text_bert` token triplets with the pinned tokenizer. Re-encoding with the same frozen local BERT produced [728,50,768] states. No model weight was changed and no A3 sample was opened.", "",
             f"Maximum absolute difference from the frozen clean BERT cache: **{max_state_diff:.8g}**.", "",
             "On the first 32 valid rows, fixed seed1729 M0/M1 checkpoints gave the following differences. Padded text BERT states were also replaced by 10,000 solely in a copied input to test whether padded queries can affect the final prediction.", "",
             "| Checkpoint | Logit difference: raw text vs cache | Intensity difference | Logit difference: padding stress | Intensity difference |",
             "|---|---:|---:|---:|---:|"]
    for level in ("M0", "M1"):
        r = probe["fixed_checkpoint_results"][level]
        lines.append(f"| {level} | {r['max_abs_logit_difference_raw_text_vs_cache']:.8g} | {r['max_abs_intensity_difference_raw_text_vs_cache']:.8g} | {r['max_abs_logit_difference_padding_stress']:.8g} | {r['max_abs_intensity_difference_padding_stress']:.8g} |")
    lines += ["", f"Padded positions in the 32-row probe batch: {probe['padded_positions_in_probe_batch']}.", "",
              "This validates an A2 shadow path `raw_text → same tokenizer → same BERT → same model output` and shows that query-side padding computation does not influence these fixed checkpoints. The strict supplied-A3-`text_bert` requirement remains unresolved for the unaligned version, and there is still no A3 inference runner. The earlier [pitfall audit](pitfall_audit.md) is preserved as the review boundary.", ""]
    (OUT / "interface_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"result": probe["result"], "max_state_diff": max_state_diff,
                      "models": probe["fixed_checkpoint_results"]}))


if __name__ == "__main__":
    main()
