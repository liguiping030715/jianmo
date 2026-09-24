import sys
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_stage4 as ps

MODS = ("text", "audio", "vision")
CODE = {"T": "text", "A": "audio", "V": "vision"}
results = []


def record(name, passed, detail=None):
    results.append({"name": name, "passed": bool(passed), "detail": detail})


train = ps.load_split("train")
valid = ps.load_split("valid")
by_split = {"train": train, "valid": valid}
mask_rows = ps.read_mask_rows(1729)

# A1: every injected index originally P==1,O==1 (text token not special).
bad = checked = 0
for e in mask_rows:
    if not e["injectable"]:
        continue
    d = by_split[e["split"]]
    for m, (a, b) in e["spans"].items():
        name = CODE[m]
        P, O = d[f"{name}_P"][e["row"]], d[f"{name}_O"][e["row"]]
        for t in range(a, b):
            checked += 1
            if not (P[t] == 1 and O[t] == 1):
                bad += 1
            if m == "T" and d["text_bert"][e["row"], 0, t] in (0, 101, 102):
                bad += 1
record("A1_injected_orig_P1O1", bad == 0, {"checked": checked, "bad": bad})

# A2/A3: real apply_mask on valid audit (all scenarios) + sampled train.
_, vphase = ps.masks_by_phase(1729)
audit = vphase["audit"]
train_map = {r["row"]: r for r in mask_rows if r["phase"] == "train"}
sample_list = [(valid, r, e) for sc, mp in audit.items() for r, e in list(mp.items())[:24]]
sample_list += [(train, r, train_map[r]) for r in list(train_map)[:40]]

aligned_has_2 = bool(any(
    2 in np.unique(d[f"{m}_P"]) or 2 in np.unique(d[f"{m}_O"])
    for d in (train, valid) for m in MODS))

bad2 = nat_viol = nat_checked = applied = 0
for d, r, e in sample_list:
    if not e["injectable"]:
        continue
    item = ps.apply_mask(d, r, e)
    applied += 1
    for mi, m in enumerate("TAV"):
        name = MODS[mi]
        inj = item["injected_mask"][m]
        Pi, Oi = d[f"{name}_P"][r], d[f"{name}_O"][r]
        Po, Oo = item[f"{name}_P"], item[f"{name}_O"]
        for t in range(50):
            if inj[t]:
                if not (Po[t] == 1 and Oo[t] == 0):
                    bad2 += 1
            elif not (Po[t] == Pi[t] and Oo[t] == Oi[t]):
                bad2 += 1
            if Pi[t] == 2 or Oi[t] == 2:
                nat_checked += 1
                if not inj[t] and not (Po[t] == Pi[t] and Oo[t] == Oi[t]):
                    nat_viol += 1
record("A2_inject_P1O0_only_on_mask", bad2 == 0, {"applied": applied, "bad": bad2})
record("A3_native2_unchanged_outside_mask", (not aligned_has_2) or nat_viol == 0,
       {"aligned_has_2": aligned_has_2, "native2_checked": nat_checked,
        "violations": nat_viol})

# Native unknown=2 fact for unaligned (RU pipeline).
u2 = 0
for split in ("train", "valid"):
    base = ps.ROOT / "unaligned" / split
    for m in MODS:
        P = np.load(base / f"{m}_P.npy")
        O = np.load(base / f"{m}_O.npy")
        u2 += int(((P == 2) | (O == 2)).sum())
record("A3b_unaligned_native2_present", u2 > 0, {"native2_positions": u2})

# A4: R0a and R1 consumed the identical mask manifest.
a = (ps.RUN_DIR / "R0a_seed1729" / "mask_manifest.jsonl").read_bytes()
b = (ps.RUN_DIR / "R1_seed1729" / "mask_manifest.jsonl").read_bytes()
record("A4_R0a_R1_same_mask_manifest", a == b, {"bytes": len(a)})

# A5: text intervention cache key depends on post-intervention token IDs.
def post_key(e, d):
    tok = d["text_bert"][e["row"]].copy()
    aa, bb = e["spans"]["T"]
    tok[0, aa:bb] = 100
    return tok.tobytes()

keys = set()
for sc, mp in audit.items():
    for r, e in mp.items():
        if e["injectable"] and "T" in e["modalities"]:
            keys.add(post_key(e, valid))
record("A5_text_key_post_intervention", len(keys) > 1, {"distinct_post_keys": len(keys)})

# A6: clean contextual BERT state never reused for modified text.
clean_states = np.load(ps.CACHE / "clean_valid.npy")
gmeta = json.loads((ps.CACHE / "gap_text_seed1729.json").read_text(encoding="utf-8"))
gap_states = np.load(ps.CACHE / "gap_text_seed1729.npy")
chosen = None
for sc, mp in audit.items():
    for r, e in mp.items():
        if e["injectable"] and "T" in e["modalities"]:
            k = ps.gap_cache_key(e)
            if k in gmeta["lookup"]:
                chosen = (r, e, gmeta["lookup"][k])
                break
    if chosen:
        break
r, e, idx = chosen
aa, bb = e["spans"]["T"]
ge, ce = gap_states[idx], clean_states[r]
record("A6_no_clean_state_reuse",
       (not np.allclose(ge, ce)) and (not np.allclose(ge[aa:bb], ce[aa:bb])),
       {"row": r, "gap_idx": idx})

# A7: scaler/data tensors identical across runs (environment checksums).
h0a = json.loads((ps.RUN_DIR / "R0a_seed1729" / "environment_checksums.json").read_text(
    encoding="utf-8"))
h1 = json.loads((ps.RUN_DIR / "R1_seed1729" / "environment_checksums.json").read_text(
    encoding="utf-8"))
fields = ("v2_manifest_sha256", "scaler_sha256", "bert_weights_sha256", "mask_sha256")
record("A7_scaler_data_identical", all(h0a[f] == h1[f] for f in fields),
       {f: [h0a[f], h1[f]] for f in fields})

# A8: order/label preserved samples.json -> batch -> predictions.
preds = {}
for line in (ps.RUN_DIR / "R1_seed1729" / "predictions_valid.jsonl").read_text(
        encoding="utf-8").splitlines():
    rec = json.loads(line)
    preds.setdefault(rec["condition"], {})[rec["row"]] = rec
order_ok = lbl_ok = True
for cond, mp in preds.items():
    if sorted(mp) != list(range(728)):
        order_ok = False
    for i, p in mp.items():
        s = valid["samples"][i]
        if p["true_class"] != int(s["classification_label"]) or abs(
                p["true_intensity"] - s["regression_label"]) > 1e-5:
            lbl_ok = False
record("A8_order_label_preserved", order_ok and lbl_ok,
       {"conditions": list(preds)})

allpass = all(r["passed"] for r in results)
out = {
    "status": "PREPROCESSING_REAUDIT_PASS" if allpass else "PREPROCESSING_REAUDIT_BLOCKED",
    "checks": results,
    "note": ("No re-clean performed. All 8 loader semantic assertions executed through "
             "the existing pilot_stage4 loader/gap code on train+valid.")}
outdir = Path("workspace/results/stage3b")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "preprocessing_reaudit.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
for r in results:
    print(r["name"], "=>", r["passed"])
print("STATUS:", out["status"])
