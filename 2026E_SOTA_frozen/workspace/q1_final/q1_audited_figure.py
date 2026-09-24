"""Plot audited Q1 alignment from saved maps; no feature extractor is rerun."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["text.parse_math"] = False

HERE = Path(__file__).resolve().parent
RID = "-tANM6ETl_M$_$3"
sm = json.loads((HERE / "source_maps" / f"{RID}.json").read_text(encoding="utf-8"))
D = sm["duration_s"]
audio_end = sm["audio_pcm_end_s"]
words = sm["words"]
fig, axes = plt.subplots(4, 1, figsize=(17, 11),
                         gridspec_kw={"height_ratios": [1.5, 1.1, 0.9, 1.65]})
fig.suptitle(f"Q1 audited source alignment: {RID}  |  edited presentation D={D:.6f} s",
             fontsize=15, weight="bold")
fig.text(0.095, 0.94, f"Official transcript: {sm['transcript']}", fontsize=9.3)

ax = axes[0]
ax.set_title("Official transcript preserved; blue = verified word timing", loc="left", fontsize=11)
for w in words:
    if w["align_status"] == "VERIFIED":
        s, e = w["start_s"], w["end_s"]
        ax.add_patch(Rectangle((s, 0.15), e - s, 0.58, facecolor="#4C72B0", alpha=0.82))
        ax.text((s + e) / 2, 0.44, w["surface"], ha="center", va="center", fontsize=7.4)
ax.set_ylim(0, 1)
ax.set_yticks([])
unresolved = [w for w in words if w["align_status"] == "UNRESOLVED"]
unresolved_text = ", ".join(f"{w['surface']} [{w['char_start']}:{w['char_end']}]" for w in unresolved)
ax.text(0.0, -0.18,
        f"UNRESOLVED words (content retained; no verified seconds): {unresolved_text}",
        transform=ax.transAxes, fontsize=10, color="#AE6B00", va="top")
ax.set_xlabel("")

ax = axes[1]
ax.set_title("Audio: 656 openSMILE LLD frames; ticks mark frame START, not frame END", loc="left", fontsize=11)
ax.axvspan(0, audio_end, color="#DCEEE4", alpha=0.8)
ax.axvspan(audio_end, D, facecolor="#EEEEEE", hatch="///", edgecolor="#888888", alpha=0.7)
ax.vlines([f["start_s"] for f in sm["audio_lld_frames"]], 0.18, 0.76,
          color="#55A868", linewidth=0.5)
last = sm["audio_lld_frames"][-1]
ax.axvline(last["end_s"], color="#267845", linestyle="--", linewidth=1.5)
ax.text(0.01, 0.94,
        f"last LLD frame [{last['start_s']:.3f}, {last['end_s']:.6f}] s; decoded PCM end={audio_end:.6f} s; tail={D-audio_end:.6f} s",
        transform=ax.transAxes, fontsize=9.4, va="top")
ax.set_ylim(0, 1)
ax.set_yticks([])

ax = axes[2]
ax.set_title("Video: 200 decoded frame PTS (candidate frames for each bin)", loc="left", fontsize=11)
ax.vlines(sm["video_decoded_frame_pts_s"], 0.14, 0.84, color="#C44E52", linewidth=0.7)
ax.set_ylim(0, 1)
ax.set_yticks([])

ax = axes[3]
ax.set_title("50 common bins: separate P and O states; 1=verified present/usable, 2=unresolved, 0=verified absent", loc="left", fontsize=10)
mods = ["text", "audio", "vision"]
colors = {"text": "#4C72B0", "audio": "#55A868", "vision": "#C44E52"}
for row, (m, field) in enumerate((m, field) for m in mods for field in ("P", "O")):
    for b in sm["bins"]:
        v = b[field][m]
        color = "#E6E6E6" if v == 0 else (colors[m] if v == 1 else "#F2B84B")
        ax.add_patch(Rectangle((b["start_s"], row), b["end_s"] - b["start_s"], 0.91,
                               color=color, alpha=0.85 if field == "O" else 0.42,
                               ec="white", lw=0.15))
ax.set_ylim(6, 0)
ax.set_yticks([i + 0.45 for i in range(6)])
ax.set_yticklabels([f"{m} {s}" for m in mods for s in ("P", "O")])
ax.axvline(audio_end, color="#267845", linestyle="--", linewidth=1)
ax.text(0.0, -0.22,
        "P=verified timeline position; O=usable local observation. Text O=1 means at least one verified word, not exhaustive word coverage.\n"
        "Text O=2 means local word location unresolved; official transcript/BERT content still exists. Audio bin 49 is P=0/O=0.",
        transform=ax.transAxes, fontsize=9.4, va="top")

for i, ax in enumerate(axes):
    ax.set_xlim(0, D)
    ax.set_xlabel("clip-relative seconds" if i else "")
fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.85, left=0.095, right=0.985)
out = HERE / "figures" / "typical_alignment_audited.png"
fig.savefig(out, dpi=150)
print(out)
