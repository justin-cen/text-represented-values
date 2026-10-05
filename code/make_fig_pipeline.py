# -*- coding: utf-8 -*-
"""论文B 方法管线示意图 figB0_pipeline.png"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

fig, ax = plt.subplots(figsize=(13.5, 4.4), dpi=200)
ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.4); ax.axis("off")

steps = [
    ("Frozen bilingual\nlexicon", "56 facets, 2,559 words", "#8C1D18"),
    ("Single multilingual\nencoder (e5)", "one shared space", "#2E5E8C"),
    ("Facet centroids\n(mean of word vectors)", "Eq. (1)", "#2E7D5B"),
    ("Cosine similarity\ntext → facet", "Eq. (2)", "#B8860B"),
    ("Within-text\nnormalization", "ipsative, Eq. (3)", "#6A3D9A"),
    ("Dimensional\ncontrast score", "pole A − pole B, Eq. (4)", "#1B6E8C"),
]
n = len(steps); bw = 1.85; gap = (13.5 - n * bw) / (n + 1)
for i, (t, sub, col) in enumerate(steps):
    x = gap + i * (bw + gap)
    box = FancyBboxPatch((x, 1.5), bw, 1.6, boxstyle="round,pad=0.06",
                         linewidth=1.6, edgecolor=col, facecolor=col, alpha=0.10)
    ax.add_patch(box)
    ax.text(x + bw / 2, 2.55, t, ha="center", va="center", fontsize=11.3, fontweight="bold", color=col)
    ax.text(x + bw / 2, 1.85, sub, ha="center", va="center", fontsize=10, color="#444")
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + bw, 2.3), (x + bw + gap, 2.3),
                                     arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#777"))
ax.text(0.3, 3.7, "Text-based value measurement pipeline", fontsize=15, fontweight="bold")
ax.text(0.3, 0.55, "Five steps: vectorize → locate facet centroids → measure proximity → "
                   "remove cross-lingual baseline within text → score dimensions by pole contrast.",
        fontsize=11.5, color="#555", style="italic")
fig.tight_layout()
fig.savefig(FIGS / "figB0_pipeline.png", bbox_inches="tight")
print("-> figB0_pipeline.png")
