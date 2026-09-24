"""Select a representative sample (prespecified rule) and plot alignment."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\研究生\jianmo\HuaweiCup-Codex-Final-2026E-SOTA")
Q1 = ROOT / "workspace" / "q1_final"
sys.path.insert(0, str(Q1))
from q1_text import TextModels
from q1_media import decode_media
from q1_audio import build_smile, audio_features
from q1_visual import build_face, visual_features

rows = json.loads((Q1 / "all_provenance.json").read_text(encoding="utf-8"))
D = np.array([r["verified_duration_s"] for r in rows])
Dmed = float(np.median(D))

cands = []
for r in rows:
    if r["text_alignment"]["status"] != "VERIFIED":
        continue
    cov = (r["bin_O_counts"]["text"] + r["bin_O_counts"]["audio"]
           + r["bin_O_counts"]["vision"]) / 150.0
    if 0.3 <= cov <= 0.95:  # exclude extreme coverage
        cands.append((abs(r["verified_duration_s"] - Dmed), cov, r))
cands.sort(key=lambda x: x[0])
_, cov, r = cands[0]
rid = r["id"]
print("typical sample:", rid, "D", r["verified_duration_s"], "cov", round(cov, 3))

# Recompute fine-grained timings.
models = TextModels()
media = decode_media(str(ROOT / "workspace" / "data" / "raw" / r["mp4"]))
words = models.word_embeddings(r["text"])
models.forced_align(words, media["pcm16_16k_mono"])
smile, _ = build_smile()
au = audio_features(smile, media["pcm16_16k_mono"])
detect, mesh = build_face()
vi = visual_features(detect, mesh, media["video_rgb"], media["video_pts"])
DD = media["duration_s"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.parse_math"] = False
from matplotlib.patches import Rectangle
fig, axes = plt.subplots(5, 1, figsize=(15, 11),
                         gridspec_kw={"height_ratios": [0.9, 1.4, 0.8, 1.1, 1.3]})

# Row 0: transcript
axes[0].axis("off")
import textwrap
axes[0].text(0.0, 0.9, f"{rid}   D={DD:.3f}s", fontsize=12, weight="bold")
axes[0].text(0.0, 0.45, textwrap.fill(r["text"], 110), fontsize=11, va="top")

# Row 1: word timing
axes[1].set_title("Text: forced-aligned word intervals (wav2vec2 CTC)")
for i, w in enumerate(words):
    if w["align_status"] == "VERIFIED":
        axes[1].add_patch(Rectangle((w["start_s"], 0.2), w["end_s"] - w["start_s"],
                                    0.6, color="#4C72B0"))
        axes[1].text((w["start_s"] + w["end_s"]) / 2, 0.5, w["word"],
                     ha="center", va="center", fontsize=8, rotation=0)
axes[1].set_ylim(0, 1); axes[1].set_yticks([])

# Row 2: audio timeline
axes[2].set_title("Audio: 16k mono timeline + eGeMAPS LLD frames")
axes[2].barh([0.5], [DD], height=0.4, color="#DDDDDD")
axes[2].vlines(au["frame_start"], 0.3, 0.7, color="#55A868", lw=0.6)
axes[2].set_ylim(0, 1); axes[2].set_yticks([])
axes[2].text(0.01, 0.85, f"{au['n_frames']} LLD frames, dim={au['dim']}",
             transform=axes[2].transAxes, fontsize=9)

# Row 3: video frames
axes[3].set_title("Video: decoded frames by PTS (green=face, grey=no face)")
for i in range(vi["n_frames"]):
    c = "#55A868" if vi["status"][i] == 1 else "#BBBBBB"
    axes[3].vlines(vi["pts"][i], 0.2, 0.8, color=c, lw=0.7)
axes[3].set_ylim(0, 1); axes[3].set_yticks([])
axes[3].text(0.01, 0.85,
             f"{vi['n_frames']} frames, faces={int(vi['status'].sum())}",
             transform=axes[3].transAxes, fontsize=9)

# Row 4: 50 bins availability for T/A/V
axes[4].set_title("50 common temporal bins: P (light) / O (solid) per modality")
band = {"text": 2, "audio": 1, "vision": 0}
colors = {"text": "#4C72B0", "audio": "#55A868", "vision": "#C44E52"}
d = dict(np.load(Q1 / "features" / f"{rid.replace('/','_')}.npz"))
K = 50
for m, b in band.items():
    P, O = d[f"{m}_P"], d[f"{m}_O"]
    for k in range(K):
        x0 = k * DD / K
        w = DD / K
        if P[k]:
            axes[4].add_patch(Rectangle((x0, b + 0.1), w * 0.95, 0.25,
                                        color=colors[m], alpha=0.3))
        if O[k]:
            axes[4].add_patch(Rectangle((x0, b + 0.4), w * 0.95, 0.5,
                                        color=colors[m], alpha=0.85))
axes[4].set_yticks([0.5, 1.5, 2.5])
axes[4].set_yticklabels(["vision", "audio", "text"])
axes[4].set_ylim(0, 3)

for ax in axes[1:]:
    ax.set_xlim(0, DD)
    ax.set_xlabel("time (s)")
fig.tight_layout()
out = Q1 / "figures" / "typical_alignment.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
