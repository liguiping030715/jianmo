"""Q3 Stage 2 — explanation result analysis and paper figures.

Reads frozen Q3 Stage 1 evidence (no model rerun, no test access).
Generates paper figures and a machine-readable summary.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("D:/研究生/jianmo/HuaweiCup-Codex-Final-2026E-SOTA")
SRC = BASE / "workspace/results/q3_stage1"
OUT = BASE / "workspace/results/q3_stage2"
FIG = BASE / "workspace/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

MODS = ["text", "audio", "vision"]
MOD_LABEL = {"text": "Text", "audio": "Audio", "vision": "Vision"}
COLORS = {"text": "#2563eb", "audio": "#16a34a", "vision": "#dc2626"}


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ── 1. Modality contribution ──────────────────────────────────────────────
print("Loading modality contribution ...")
mc = load_csv(SRC / "q3_modality_contribution.csv")
n = len(mc)

delta_cls = {m: np.array([float(r[f"Delta_cls_{m}"]) for r in mc]) for m in MODS}
delta_reg = {m: np.array([float(r[f"Delta_reg_{m}"]) for r in mc]) for m in MODS}
dom_cls = Counter(r["dominant_modality_cls"] for r in mc)
dom_reg = Counter(r["dominant_modality_reg"] for r in mc)

# joint audio+vision
delta_cls_av = np.array([float(r["Delta_cls_audio_vision_joint"]) for r in mc])
delta_reg_av = np.array([float(r["Delta_reg_audio_vision_joint"]) for r in mc])

# Fig Q3-1: boxplot of modality donor responses
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, data, title, ylabel in [
    (axes[0], delta_cls, "Classification logit drop", "Δ logit (full − donor)"),
    (axes[1], delta_reg, "Regression intensity change", "|Δ intensity|"),
]:
    bp = ax.boxplot(
        [data[m] for m in MODS],
        labels=[MOD_LABEL[m] for m in MODS],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, m in zip(bp["boxes"], MODS):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"Q3 — Modality donor replacement response (n={n}, frozen M0 seed2718)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "q3_modality_contribution_boxplot.png")
plt.close(fig)
print("  saved q3_modality_contribution_boxplot.png")

# Fig Q3-2: dominant modality distribution
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, counter, title in [
    (axes[0], dom_cls, "Dominant modality — classification"),
    (axes[1], dom_reg, "Dominant modality — regression"),
]:
    labels = [MOD_LABEL.get(m, m) for m in MODS]
    vals = [counter.get(m, 0) for m in MODS]
    pcts = [v / n * 100 for v in vals]
    bars = ax.bar(labels, pcts, color=[COLORS[m] for m in MODS], alpha=0.75)
    for bar, v, p in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v}\n({p:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Samples (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(pcts) * 1.2)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Q3 — Dominant contributing modality per sample", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "q3_dominant_modality.png")
plt.close(fig)
print("  saved q3_dominant_modality.png")

# ── 2. Faithfulness ───────────────────────────────────────────────────────
print("Loading faithfulness ...")
faith = json.loads((SRC / "q3_faithfulness.json").read_text(encoding="utf-8"))

faith_rows = []
for m in MODS:
    for target in ["cls", "reg"]:
        entry = faith["faithfulness"][m][target]
        jd = entry["joint_deletion"]
        ptr = entry.get("paired_top_minus_random", {})
        prb = entry.get("paired_random_minus_bottom", {})
        faith_rows.append({
            "modality": m, "target": target,
            "eligible": entry["eligible_samples"],
            "top_mean": jd["top"]["mean"],
            "random_mean": jd["random"]["mean"],
            "bottom_mean": jd["bottom"]["mean"],
            "top_random_ci": str(ptr.get("ci", "")),
            "random_bottom_ci": str(prb.get("ci", "")),
            "pass": entry["passes_operator_consistency"],
        })

# Fig Q3-3: faithfulness top/random/bottom
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey="row")
for row, target in enumerate(["cls", "reg"]):
    for col, m in enumerate(MODS):
        ax = axes[row][col]
        jd = faith["faithfulness"][m][target]["joint_deletion"]
        labels = ["Top", "Random", "Bottom"]
        means = [jd["top"]["mean"], jd["random"]["mean"], jd["bottom"]["mean"]]
        bars = ax.bar(labels, means, color=["#dc2626", "#9ca3af", "#2563eb"], alpha=0.75, width=0.55)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{MOD_LABEL[m]} — {'Classification' if target=='cls' else 'Regression'}")
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("Joint deletion response")
fig.suptitle("Q3 — Faithfulness: top > random > bottom joint deletion (frozen operator)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "q3_faithfulness_top_random_bottom.png")
plt.close(fig)
print("  saved q3_faithfulness_top_random_bottom.png")

# ── 3. Stability ──────────────────────────────────────────────────────────
stab_rows = []
for m in MODS:
    for target in ["cls", "reg"]:
        entry = faith["stability"][m][target]
        stab_rows.append({
            "modality": m, "target": target,
            "prediction_preserving_rows": entry["eligible_rows"],
            "top1_agreement": entry["top1_agreement"],
            "top3_jaccard": entry["mean_top3_jaccard"],
        })

# Fig Q3-4: stability
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(MODS))
w = 0.35
cls_t1 = [faith["stability"][m]["cls"]["top1_agreement"] for m in MODS]
reg_t1 = [faith["stability"][m]["reg"]["top1_agreement"] for m in MODS]
cls_t3 = [faith["stability"][m]["cls"]["mean_top3_jaccard"] for m in MODS]
reg_t3 = [faith["stability"][m]["reg"]["mean_top3_jaccard"] for m in MODS]
ax.bar(x - w/2, cls_t1, w, label="Cls top-1", color="#2563eb", alpha=0.7)
ax.bar(x + w/2, reg_t1, w, label="Reg top-1", color="#dc2626", alpha=0.7)
ax.plot(x, cls_t3, "o--", color="#1e40af", label="Cls top-3 Jaccard", alpha=0.8)
ax.plot(x, reg_t3, "s--", color="#991b1b", label="Reg top-3 Jaccard", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([MOD_LABEL[m] for m in MODS])
ax.set_ylabel("Agreement / Jaccard")
ax.set_ylim(0, 1.08)
ax.set_title("Q3 — Explanation stability under small perturbation (prediction-preserving rows)")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "q3_stability.png")
plt.close(fig)
print("  saved q3_stability.png")

# ── 4. Local text evidence — top phrases ──────────────────────────────────
print("Loading local text evidence ...")
lt = load_csv(SRC / "q3_local_text_evidence.csv")
# use scale=3 (3-token) candidates, rank by abs Delta_cls
lt3 = [r for r in lt if r["scale"] == "3"]
lt3_sorted = sorted(lt3, key=lambda r: abs(float(r["Delta_cls"])), reverse=True)

# aggregate by phrase_text (mean abs Delta_cls, count)
phrase_agg = defaultdict(list)
for r in lt3:
    phrase_agg[r["phrase_text"].strip().lower()].append(abs(float(r["Delta_cls"])))
phrase_stats = sorted(
    [(p, np.mean(v), len(v)) for p, v in phrase_agg.items() if len(v) >= 3],
    key=lambda x: x[1], reverse=True,
)

# Fig Q3-5: top phrases by mean |Delta_cls|
top_n = 20
top_phrases = phrase_stats[:top_n]
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = [p[0] if p[0] else "(empty)" for p in top_phrases][::-1]
means = [p[1] for p in top_phrases][::-1]
counts = [p[2] for p in top_phrases][::-1]
bars = ax.barh(range(len(labels)), means, color="#2563eb", alpha=0.7)
for i, (bar, c) in enumerate(zip(bars, counts)):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"n={c}", va="center", fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Mean |Δ classification logit| (3-token deletion)")
ax.set_title(f"Q3 — Top text phrases by classification impact (3-token, n≥3 occurrences)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "q3_top_text_phrases.png")
plt.close(fig)
print("  saved q3_top_text_phrases.png")

# ── 5. Local audio/vision evidence summary ────────────────────────────────
print("Loading local audio/vision evidence ...")
la = load_csv(SRC / "q3_local_audio_evidence.csv")
lv = load_csv(SRC / "q3_local_vision_evidence.csv")

def local_summary(rows, name):
    by_scale = defaultdict(lambda: {"cls": [], "reg": []})
    for r in rows:
        s = r["scale"]
        by_scale[s]["cls"].append(abs(float(r["Delta_cls"])))
        by_scale[s]["reg"].append(abs(float(r["Delta_reg"])))
    out = []
    for s in sorted(by_scale.keys(), key=lambda x: float(x)):
        out.append({
            "modality": name, "scale_pct": s,
            "mean_abs_delta_cls": np.mean(by_scale[s]["cls"]),
            "mean_abs_delta_reg": np.mean(by_scale[s]["reg"]),
            "n": len(by_scale[s]["cls"]),
        })
    return out

av_local = local_summary(la, "audio") + local_summary(lv, "vision")

# ── 6. Write summary CSVs ─────────────────────────────────────────────────
with open(OUT / "q3_modality_contribution_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["modality", "target", "n", "mean", "median", "p05", "p95", "min", "max"])
    for m in MODS:
        for target, arr in [("cls", delta_cls[m]), ("reg", delta_reg[m])]:
            w.writerow([m, target, len(arr),
                        f"{np.mean(arr):.6f}", f"{np.median(arr):.6f}",
                        f"{np.percentile(arr,5):.6f}", f"{np.percentile(arr,95):.6f}",
                        f"{np.min(arr):.6f}", f"{np.max(arr):.6f}"])
    w.writerow(["audio_vision_joint", "cls", len(delta_cls_av),
                f"{np.mean(delta_cls_av):.6f}", f"{np.median(delta_cls_av):.6f}",
                f"{np.percentile(delta_cls_av,5):.6f}", f"{np.percentile(delta_cls_av,95):.6f}",
                f"{np.min(delta_cls_av):.6f}", f"{np.max(delta_cls_av):.6f}"])
    w.writerow(["audio_vision_joint", "reg", len(delta_reg_av),
                f"{np.mean(delta_reg_av):.6f}", f"{np.median(delta_reg_av):.6f}",
                f"{np.percentile(delta_reg_av,5):.6f}", f"{np.percentile(delta_reg_av,95):.6f}",
                f"{np.min(delta_reg_av):.6f}", f"{np.max(delta_reg_av):.6f}"])

with open(OUT / "q3_dominant_modality_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["target", "text", "audio", "vision", "total"])
    w.writerow(["cls", dom_cls.get("text",0), dom_cls.get("audio",0), dom_cls.get("vision",0), n])
    w.writerow(["reg", dom_reg.get("text",0), dom_reg.get("audio",0), dom_reg.get("vision",0), n])

with open(OUT / "q3_faithfulness_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(faith_rows[0].keys()))
    w.writeheader(); w.writerows(faith_rows)

with open(OUT / "q3_stability_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(stab_rows[0].keys()))
    w.writeheader(); w.writerows(stab_rows)

with open(OUT / "q3_top_text_phrases.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phrase", "mean_abs_delta_cls", "count"])
    for p, m, c in phrase_stats[:50]:
        w.writerow([p, f"{m:.6f}", c])

with open(OUT / "q3_local_av_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(av_local[0].keys()))
    w.writeheader(); w.writerows(av_local)

# ── 7. Representative case study ──────────────────────────────────────────
# pick 3 samples: one text-dominant, one audio-dominant, one vision-dominant
print("Selecting representative cases ...")
cases = {}
for target_mod in MODS:
    candidates = [r for r in mc if r["dominant_modality_cls"] == target_mod]
    if candidates:
        # pick the one with highest Delta_cls for that modality
        best = max(candidates, key=lambda r: float(r[f"Delta_cls_{target_mod}"]))
        cases[target_mod] = best

with open(OUT / "q3_representative_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case_type", "sample_id", "predicted_class", "predicted_polarity",
                 "full_logit", "full_intensity",
                 "Delta_cls_text", "Delta_cls_audio", "Delta_cls_vision",
                 "Delta_reg_text", "Delta_reg_audio", "Delta_reg_vision",
                 "dominant_cls", "dominant_reg"])
    for m, r in cases.items():
        w.writerow([f"{m}_dominant", r["sample_id"], r["predicted_class_index"],
                     r["predicted_polarity"], r["full_predicted_class_logit"],
                     r["full_intensity"],
                     r["Delta_cls_text"], r["Delta_cls_audio"], r["Delta_cls_vision"],
                     r["Delta_reg_text"], r["Delta_reg_audio"], r["Delta_reg_vision"],
                     r["dominant_modality_cls"], r["dominant_modality_reg"]])

# ── 8. Stage 2 status ─────────────────────────────────────────────────────
status = {
    "status": "Q3_STAGE2_ANALYSIS_COMPLETE",
    "scope": "Inference-only analysis of frozen Q3 Stage 1 evidence. No model update, no A2 test / Attachment 3 / 4 access.",
    "n_samples": n,
    "frozen_checkpoint_sha256": faith["checkpoint_sha256"],
    "figures": [
        "q3_modality_contribution_boxplot.png",
        "q3_dominant_modality.png",
        "q3_faithfulness_top_random_bottom.png",
        "q3_stability.png",
        "q3_top_text_phrases.png",
    ],
    "key_findings": {
        "text_is_dominant_modality": f"text dominant in cls: {dom_cls.get('text',0)}/{n} ({dom_cls.get('text',0)/n*100:.1f}%), reg: {dom_reg.get('text',0)}/{n} ({dom_reg.get('text',0)/n*100:.1f}%)",
        "text_donor_response_much_larger": f"text mean Delta_cls={np.mean(delta_cls['text']):.4f} vs audio={np.mean(delta_cls['audio']):.4f} vs vision={np.mean(delta_cls['vision']):.4f}",
        "faithfulness_all_pass": all(r["pass"] for r in faith_rows),
        "text_stability_lower_than_av": f"text cls top1={faith['stability']['text']['cls']['top1_agreement']:.3f}, audio cls top1={faith['stability']['audio']['cls']['top1_agreement']:.3f}",
        "audio_vision_joint_less_than_text": f"joint A+V mean Delta_cls={np.mean(delta_cls_av):.4f} < text={np.mean(delta_cls['text']):.4f}",
    },
    "limitations": [
        "Donor responses are model intervention responses, not causal effects.",
        "Text phrases are tokenizer reconstructions, not verified spoken timestamps.",
        "Audio/vision use native indices only; seconds are unavailable.",
        "Stability perturbation is a declared small masking/[UNK] intervention, not a comprehensive robustness test.",
        "No attention-weight attribution or certified hidden missing mask is used.",
    ],
}
(OUT / "q3_stage2_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n=== Q3 Stage 2 complete ===")
print(f"  n={n}, figures={len(status['figures'])}, CSVs=7")
print(f"  text dominant cls: {dom_cls.get('text',0)}/{n} ({dom_cls.get('text',0)/n*100:.1f}%)")
print(f"  text dominant reg: {dom_reg.get('text',0)}/{n} ({dom_reg.get('text',0)/n*100:.1f}%)")
print(f"  faithfulness all pass: {all(r['pass'] for r in faith_rows)}")
