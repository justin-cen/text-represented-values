# -*- coding: utf-8 -*-
"""生成论文插图（中文宋体系字体）：
图1 研究框架图；图2 六层×六维对比分折线；图3 分媒体 IND×LTO 散点。
输出到 work/paper/figs/
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"E:\culture-difference\work\paper\figs")
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================== 图1 研究框架图 ==============================
fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, text, fc="#EEF4FB", ec="#3B6EA5", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color="#555555", lw=1.2))


box(0.2, 3.3, 2.5, 1.3, "同片影评\n豆瓣×Letterboxd\n717部 / 8,968条", fc="#FBF0E8", ec="#C0703F")
box(0.2, 0.7, 2.5, 1.3, "月份对齐新闻\n人民日报系×西方媒体\n27,000篇", fc="#FBF0E8", ec="#C0703F")
box(3.9, 2.0, 2.7, 1.6, "冻结双语词库\n4体系·56刻面·2,559词\nLLM扩充＋机器筛查", fc="#EAF6EC", ec="#4E8A54")
box(7.1, 3.3, 2.7, 1.3, "统一嵌入测量\n多语e5向量空间\n文本内标准化＋对比分", fc="#F1ECF8", ec="#7A5FA8")
box(7.1, 0.7, 2.7, 1.3, "三层验证\n信度·稳健性·效标关联", fc="#F1ECF8", ec="#7A5FA8")
arrow(2.7, 3.95, 3.9, 3.2)
arrow(2.7, 1.35, 3.9, 2.4)
arrow(6.6, 2.8, 7.1, 3.95)
arrow(6.6, 2.8, 7.1, 1.35)
ax.set_title("图1 研究框架", fontsize=11, pad=10)
fig.tight_layout()
fig.savefig(OUT / "fig1_framework.png", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ============================== 图2 十一层×六维折线 ==============================
layers = ["新闻·历史\n05-11/10-17", "新闻·近期\n2026/2025", "新闻·窗口\n26-08/25-08",
          "影评·历史\n豆瓣/IMDB", "豆瓣×IMDB\n≤11/2011", "豆瓣×genome\n≤12/2012",
          "影评·西片\n豆瓣/IMDB", "影评·同片\n豆瓣19/LB25", "同片同时代\n豆瓣20-26/LB25", "华语片\n豆瓣/LB25",
          "政治·国家层\nGWR/西方施政"]
data = {
    "IDV": [-2.30, 1.81, 1.31, -0.56, 2.23, 1.11, -1.01, 0.73, 0.30, 1.47, -1.83],
    "PDI": [0.78, -2.33, -2.40, 0.77, 1.03, 2.05, 0.81, 2.04, 2.25, 2.51, 2.09],
    "MAS": [2.94, 3.72, 3.96, 0.18, -1.20, 0.08, -0.41, 0.77, 2.04, 2.38, 6.36],
    "UAI": [6.56, 4.78, 4.52, 5.60, 5.56, 5.21, 5.23, 4.59, 5.14, 5.22, 4.69],
    "LTO": [-13.27, -9.23, -9.32, -8.47, -7.99, -9.81, -7.40, -9.78, -10.34, -11.46, -7.84],
    "IND": [-17.45, -21.13, -20.63, -15.63, -17.29, -17.07, -14.50, -15.15, -14.82, -16.14, -14.71],
}
style = {
    "IDV": dict(c="#999999", m="o", ls="--"),
    "PDI": dict(c="#1f77b4", m="s", ls="--"),
    "MAS": dict(c="#2ca02c", m="^", ls="-."),
    "UAI": dict(c="#9467bd", m="D", ls="-"),
    "LTO": dict(c="#d62728", m="v", ls="-"),
    "IND": dict(c="#ff7f0e", m="o", ls="-"),
}
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
x = np.arange(len(layers))
for dim, vals in data.items():
    st = style[dim]
    ax.plot(x, vals, color=st["c"], marker=st["m"], ls=st["ls"], lw=1.6,
            ms=5.5, label=dim)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=8.5)
ax.set_ylabel("维度对比分（zh-west, ×1000）")
ax.set_title("图2 六个 Hofstede 维度对比分的跨年代×跨体裁稳定性", fontsize=11)
ax.legend(ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_layers.png", bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ============================== 图3 分媒体散点 ==============================
# (outlet: LTO, IND, side)
outlets = [
    ("人民日报", -6.0, -3.4, "zh"), ("新华网", -7.2, -5.2, "zh"), ("中新网", -8.5, -8.5, "zh"),
    ("Independent", 3.5, 17.6, "west"), ("Fox", 1.9, 17.2, "west"),
    ("Telegraph", 4.4, 18.4, "west"), ("WaPo", 2.5, 18.5, "west"),
    ("NYT", 1.5, 17.7, "west"), ("CNN", 1.4, 11.0, "west"),
    ("Reuters", 3.8, 14.8, "west"), ("AP", 1.2, 19.2, "west"),
    ("BBC", 1.7, 15.1, "west"), ("USAToday", 5.2, 22.8, "west"),
    ("Guardian", 5.5, 22.4, "west"), ("NPR", 7.6, 19.5, "west"),
]
fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200)
for name, x0, y0, side in outlets:
    c = "#C0392B" if side == "zh" else "#2E5E8C"
    m = "s" if side == "zh" else "o"
    ax.scatter(x0, y0, c=c, marker=m, s=64, zorder=3,
               edgecolors="white", linewidths=0.8)
    dy = 0.6 if name not in ("Fox", "WaPo", "NYT", "Reuters") else -1.1
    ax.annotate(name, (x0, y0), textcoords="offset points", xytext=(7, dy * 5),
                fontsize=8.2)
ax.axhline(0, color="#333333", lw=0.8)
ax.axvline(0, color="#333333", lw=0.8)
ax.set_xlabel("LTO 长期导向对比分（×1000）")
ax.set_ylabel("IND 约束←→放纵对比分（×1000）")
ax.set_title("图3 分媒体稳健性：IND 与 LTO 的中西媒体分离", fontsize=11)
ax.scatter([], [], c="#C0392B", marker="s", s=60, label="中文媒体")
ax.scatter([], [], c="#2E5E8C", marker="o", s=60, label="西方媒体")
ax.legend(fontsize=9, loc="upper left", frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_outlets.png", bbox_inches="tight")
plt.close(fig)
print("fig3 done")
print("ALL FIGURES ->", OUT)
