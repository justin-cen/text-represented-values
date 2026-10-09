# -*- coding: utf-8 -*-
"""预测效度系数图：价值观语域对评分的标准化回归系数（含情感对照）。"""
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

d = pd.read_csv(BASE / "results" / "predictive_validity.csv")
d = d[d["变量"] != "const"]
order = ["权力距离", "不确定性规避", "长/短期导向", "个体主义", "男性气质", "放纵/约束", "sentiment"]
lab_map = {"sentiment": "文本情感（对照）"}
d["label"] = d["变量"].map(lambda x: lab_map.get(x, x))
d = d.set_index("变量").loc[order].reset_index()

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
colors = ["#2E7D5B" if b > 0 else "#C0392B" for b in d["标准化系数β"]]
colors = ["#888" if v == "文本情感（对照）" else c for v, c in zip(d["label"], colors)]
ax.barh(d["label"], d["标准化系数β"], color=colors, alpha=0.9,
        xerr=d["稳健SE"] * 1.96, error_kw=dict(lw=1, capsize=3))
ax.axvline(0, color="#333", lw=0.9)
for i, (b, p) in enumerate(zip(d["标准化系数β"], d["p"])):
    ax.text(b + (0.006 if b >= 0 else -0.006), i, f"{b:+.3f}",
            va="center", ha="left" if b >= 0 else "right", fontsize=11.5)
ax.set_xlabel("标准化回归系数 β（对评分的效应，±1.96·SE）")
ax.set_title("影评语域对观众评分的预测效应（n=3,134 部影片，控制情感/片长/热度，增量 R²=+8.4%）", fontsize=14, pad=12)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "figC_predictive.png", bbox_inches="tight")
print("-> figC_predictive.png")
