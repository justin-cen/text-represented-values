# -*- coding: utf-8 -*-
"""事件研究时序图：最高政治话语层价值观语域（zh−west）1990—2026，标重大事件。"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False
BASE = Path(r"E:\culture-difference\data_corpus")
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = pd.read_csv(BASE / "results" / "political_yearpair.csv")
d = d[(d.year >= 1990) & (d.year <= 2026)].reset_index(drop=True)
EVENTS = [(2001, "WTO 入世"), (2008, "北京奥运/GFC"), (2013, "SCV 正式提出"), (2020, "新冠疫情")]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=200)
show1 = [("MAS", "男性气质"), ("UAI", "不确定性规避")]
show2 = [("IND", "放纵/约束"), ("LTO", "长/短期导向")]
for ax, items, title in [(axes[0], show1, "2008 北京奥运/GFC 显著变化"),
                         (axes[1], show2, "2020 新冠显著变化")]:
    for dim, cn in items:
        ax.plot(d.year, d[dim], marker="o", ms=4.5, lw=2, label=cn)
    for y, lab in EVENTS:
        ax.axvline(y, color="#888", lw=1, ls="--", alpha=0.7)
        ax.text(y, ax.get_ylim()[1], f" {lab}", rotation=90, va="top", ha="left", fontsize=10, color="#666")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("年份")
    ax.set_ylabel("对比分（zh−west, ×1000）")
    ax.legend(frameon=False, fontsize=11.5)
    ax.grid(alpha=0.3)
axes[0].set_ylim(-2, 12)
fig.suptitle("最高政治话语层价值观语域的时间动态与重大事件标记（1990—2026）", fontsize=15.5, y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "figC_event_study.png", bbox_inches="tight")
print("-> figC_event_study.png")
