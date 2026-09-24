import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "workspace" / "experiments"))
import pilot_stage4 as ps

SEEDS = (1729, 2718, 31415)
MODNAME = ("text", "audio", "vision")
CODE = {0: "T", 1: "A", 2: "V"}

valid = ps.load_split("valid")
states = np.load(ps.CACHE / "clean_valid.npy")
samples = valid["samples"]

# Deterministic 60-sample subset: source-video grouped, class stratified.
quota = {0: 17, 1: 15, 2: 28}


def vid(s):
    return s["id"].split("$_$")[0]


selected = []
for cls, q in quota.items():
    groups = {}
    for i, s in enumerate(samples):
        if int(s["classification_label"]) == cls:
            groups.setdefault(vid(s), []).append(i)
    order = sorted(groups, key=lambda v: hashlib.sha256(("q3a2|" + v).encode()).hexdigest())
    take = []
    for v in order:
        for i in sorted(groups[v]):
            take.append(i)
            if len(take) >= q:
                break
        if len(take) >= q:
            break
    selected += take
rows = np.array(sorted(selected))
assert len(rows) == 60, len(rows)


def make_tensors(rws):
    t = states[rws].copy()
    a = valid["audio"][rws].copy()
    v = valid["vision"][rws].copy()
    p = np.stack([valid[f"{m}_P"][rws] for m in MODNAME], 1).copy()
    o = np.stack([valid[f"{m}_O"][rws] for m in MODNAME], 1).copy()
    return t, a, v, p, o


def fwd(model, t, a, v, p, o):
    with torch.inference_mode():
        z, y, _ = model(*[torch.from_numpy(x) for x in (t, a, v, p, o)])
    return F.softmax(z, -1).numpy(), y.numpy()


def span_in(idxs, length):
    for n in range(length, 0, -1):
        sp = ps.possible_spans(idxs, n)
        if sp:
            return sp[0]
    return None


allout = {"subset_n": 60, "rows": rows.tolist(), "seeds": {}}
for seed in SEEDS:
    model = ps.PilotHead("B1")
    ck = torch.load(ps.RUN_DIR / f"R1_seed{seed}" / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(ck)
    model.eval()
    t, a, v, p, o = make_tensors(rows)
    pb, rb = fwd(model, t, a, v, p, o)
    pred = pb.argmax(1)
    dels = {}
    for m in range(3):
        t2, a2, v2, p2 = t.copy(), a.copy(), v.copy(), p.copy()
        (t2, a2, v2)[m][:] = 0
        p2[:, m] = 0
        dels[m] = fwd(model, t2, a2, v2, p2, o)
    rec = []
    for k, r in enumerate(rows):
        dcls = {CODE[m]: float(pb[k, pred[k]] - dels[m][0][k, pred[k]]) for m in range(3)}
        dreg = {CODE[m]: float(rb[k] - dels[m][1][k]) for m in range(3)}
        dom = max(range(3), key=lambda m: dcls[CODE[m]])

        def del_interval(span):
            t3, a3, v3, p3, o3 = make_tensors(np.array([r]))
            aa, bb = span
            if dom == 0:
                t3[0, aa:bb] = 0
            elif dom == 1:
                a3[0, aa:bb] = 0
                o3[0, 1, aa:bb] = 0
            else:
                v3[0, aa:bb] = 0
                o3[0, 2, aa:bb] = 0
            p3[0, dom, aa:bb] = 0
            pp, rr = fwd(model, t3, a3, v3, p3, o3)
            return [float(pb[k, pred[k]] - pp[0, pred[k]]), float(rb[k] - rr[0])]

        elig = ps.eligible_positions(valid, r, CODE[dom])
        thirds = np.array_split(elig, 3)
        mspan = span_in(thirds[1], 16)
        cspan = span_in(thirds[0], (mspan[1] - mspan[0]) if mspan else 16) if mspan else None
        rec.append({
            "row": int(r), "pred": int(pred[k]),
            "true": int(samples[r]["classification_label"]),
            "delta_cls": dcls, "delta_reg": dreg, "dominant": CODE[dom],
            "interval": del_interval(mspan) if mspan else None,
            "control": del_interval(cspan) if cspan else None,
            "span_len": (mspan[1] - mspan[0]) if mspan else 0})
    dom_counts = {c: sum(x["dominant"] == c for x in rec) for c in "TAV"}
    mean_dcls = {c: float(np.mean([x["delta_cls"][c] for x in rec])) for c in "TAV"}
    mean_dreg = {c: float(np.mean([x["delta_reg"][c] for x in rec])) for c in "TAV"}
    iv = [x["interval"] for x in rec if x["interval"] is not None]
    cv = [x["control"] for x in rec if x["control"] is not None]
    allout["seeds"][str(seed)] = {
        "dominant_counts": dom_counts, "mean_delta_cls": mean_dcls,
        "mean_delta_reg": mean_dreg,
        "interval_mean": [float(np.mean([q[0] for q in iv])), float(np.mean([q[1] for q in iv]))],
        "control_mean": [float(np.mean([q[0] for q in cv])), float(np.mean([q[1] for q in cv]))],
        "per_sample": rec}

# cross-seed summary
summ = {"dominant_counts": {c: [allout["seeds"][str(s)]["dominant_counts"][c] for s in SEEDS]
                            for c in "TAV"},
        "mean_delta_cls": {c: float(np.mean([allout["seeds"][str(s)]["mean_delta_cls"][c]
                                            for s in SEEDS])) for c in "TAV"},
        "mean_delta_reg": {c: float(np.mean([allout["seeds"][str(s)]["mean_delta_reg"][c]
                                            for s in SEEDS])) for c in "TAV"},
        "interval_mean": [float(np.mean([allout["seeds"][str(s)]["interval_mean"][i]
                                        for s in SEEDS])) for i in range(2)],
        "control_mean": [float(np.mean([allout["seeds"][str(s)]["control_mean"][i]
                                       for s in SEEDS])) for i in range(2)]}
allout["cross_seed_summary"] = summ
allout["label"] = ["DIAGNOSTIC_ONLY", "PREDICTOR_NOT_PROMOTED"]

outdir = ps.RESULT
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "q3_diagnostic.json").write_text(json.dumps(allout, indent=2), encoding="utf-8")
print("dominant counts (per seed T/A/V):",
      {c: summ["dominant_counts"][c] for c in "TAV"})
print("mean dCls:", {c: round(summ["mean_delta_cls"][c], 4) for c in "TAV"})
print("mean dReg:", {c: round(summ["mean_delta_reg"][c], 4) for c in "TAV"})
print("interval [dCls,dReg]:", [round(x, 4) for x in summ["interval_mean"]])
print("control  [dCls,dReg]:", [round(x, 4) for x in summ["control_mean"]])
