# -*- coding: utf-8 -*-
"""三层共识图（论文A 核心图）：
左：三层 SCV 12 值画像对比（国家/媒体/大众）；右：层间相关矩阵热图。
输出 work/paper/figs/figA_layer_consensus.png
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"E:\culture-difference\data_corpus")
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCV_CN = ["富强", "民主", "文明", "和谐", "自由", "平等", "公正", "法治", "爱国", "敬业", "诚信", "友善"]
prof = pd.read_csv(BASE / "results" / "layer_profiles.csv", index_col=0)
scv_rows = [i for i in prof.index if str(i).startswith("SCV_")]
S = prof.loc[scv_rows, [c for c in prof.columns if c.startswith("SCV_")]].copy()
S.index = SCV_CN
corr = pd.read_csv(BASE / "results" / "layer_consensus.csv", index_col=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.85, 1]})

x = np.arange(len(SCV_CN))
w = 0.27
colors = ["#8C1D18", "#C0392B", "#2E5E8C"]
marks = ["o", "s", "^"]
cols = list(S.columns)
for k, (c, col, mk) in enumerate(zip(cols, colors, marks)):
    ax1.bar(x + (k - 1) * w, S[c].values, w, label=c.replace("SCV_", "").replace("·", " · "),
            color=col, alpha=0.88, edgecolor="white", linewidth=0.6)
ax1.axhline(0, color="#444", lw=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(SCV_CN, fontsize=12.5)
ax1.set_ylabel("相对凸显度（ipsative ×1000）", fontsize=13)
ax1.set_title("三层话语的社会主义核心价值观画像", fontsize=15, pad=12)
ax1.legend(frameon=False, fontsize=11.5, loc="lower right", ncol=1)
ax1.grid(axis="y", alpha=0.3)
# 标注敬业三层几乎一致的"共识核心"
i_ded = SCV_CN.index("敬业")
for k, col in enumerate(colors):
    v = S[cols[k]].values[i_ded]
    ax1.annotate(f"{v:+.1f}", (i_ded + (k - 1) * w, v), ha="center", va="bottom",
                 fontsize=10.5, color=col, xytext=(0, 2), textcoords="offset points")

short = [c.split("·")[-1] for c in cols]
im = ax2.imshow(corr.values, cmap="RdYlBu_r", vmin=0.5, vmax=1.0)
ax2.set_xticks(range(3)); ax2.set_xticklabels(short, fontsize=12, rotation=18, ha="right")
ax2.set_yticks(range(3)); ax2.set_yticklabels(short, fontsize=12)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                 fontsize=12.5, color="white" if corr.values[i, j] > 0.85 else "#222")
ax2.set_title("层间画像相关（SCV 12 值，Pearson r）", fontsize=15, pad=12)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIGS / "figA_layer_consensus.png", bbox_inches="tight")
print("-> figA_layer_consensus.png")
