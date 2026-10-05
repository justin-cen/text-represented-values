# -*- coding: utf-8 -*-
"""加厚图：
figA4_genre_interaction.png —— 体裁×维度交互（影评/新闻/政治三层六维对比分）
figA5_outlet_consistency.png —— 四家中文官媒 SCV 画像一致性
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
BASE = Path(r"E:\culture-difference\data_corpus\results")
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- figA4 体裁×维度交互 ----------
dims = ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]
reviews = [2.03, 0.73, 0.75, 4.53, -9.68, -15.10]
news = [-2.33, 1.81, 3.72, 4.78, -9.23, -21.13]
political = [2.04, -1.85, 6.37, 4.69, -7.89, -14.74]

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=200)
x = np.arange(len(dims)); w = 0.27
ax.bar(x - w, reviews, w, label="大众层·同片影评", color="#2E5E8C", alpha=0.9)
ax.bar(x, news, w, label="媒体层·月份新闻", color="#C0392B", alpha=0.9)
ax.bar(x + w, political, w, label="国家层·政治施政", color="#2E7D5B", alpha=0.9)
ax.axhline(0, color="#444", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(dims, fontsize=13)
ax.set_ylabel("对比分（zh−west, ×1000）", fontsize=13)
ax.set_title("体裁×维度交互：三层话语的六维对比分", fontsize=15.5, pad=12)
ax.legend(frameon=False, fontsize=12); ax.grid(axis="y", alpha=0.3)
ax.annotate("PDI：影评/政治为正、新闻为负\n（清晰的体裁效应）", xy=(0, 4), xytext=(0.6, 8.5),
            fontsize=11.5, arrowprops=dict(arrowstyle="->", color="#555"), color="#333")
fig.tight_layout()
fig.savefig(FIGS / "figA4_genre_interaction.png", bbox_inches="tight")
plt.close(fig)
print("-> figA4_genre_interaction.png")

# ---------- figA5 四家官媒一致性 ----------
x_ = pd.ExcelFile(BASE / "scv_china_side.xlsx")
zhm = x_.parse("zh_media_profile").set_index("channel")
outlets = ["人民日报", "新华网", "中新网", "央广网"]
show = ["富强", "民主", "文明", "和谐", "敬业"]
fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=200)
xx = np.arange(len(show)); w = 0.19
cols = ["#8C1D18", "#C0392B", "#D98880", "#2E5E8C"]
for k, (o, col) in enumerate(zip(outlets, cols)):
    ax.bar(xx + (k - 1.5) * w, [zhm.loc[o, v] for v in show], w, label=o, color=col, alpha=0.9)
ax.axhline(0, color="#444", lw=0.9)
ax.set_xticks(xx); ax.set_xticklabels(show, fontsize=13)
ax.set_ylabel("相对凸显度（ipsative ×1000）", fontsize=13)
ax.set_title("四家中文官方媒体的核心价值观画像（高度一致）", fontsize=15, pad=12)
ax.legend(frameon=False, fontsize=11.5, ncol=4, loc="upper center")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "figA5_outlet_consistency.png", bbox_inches="tight")
plt.close(fig)
print("-> figA5_outlet_consistency.png")
