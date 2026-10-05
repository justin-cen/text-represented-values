# -*- coding: utf-8 -*-
"""论文B 核心图：四重稳健性证据矩阵 + 构念边界。
figB1_robustness.png  —— 六维 × 五重检验的通过矩阵
figB2_construct_boundary.png —— 文本得分 vs 调查基准（不收敛）
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

dims = ["Indulgence\n-restraint", "Power\ndistance", "Masculinity", "Long-term\norientation",
        "Identity", "Uncertainty\navoidance"]
checks = ["Lexicon\nversion", "11-layer\n(within model)", "Placebo\nspecificity",
          "Cross-model\n(LaBSE)", "Item anchor\n(VSM2013)"]
# 1.0 = 通过；0.5 = 部分/体裁依赖；0.0 = 未通过
M = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0],   # IND
    [1.0, 0.5, 0.0, 0.5, 1.0],   # PDI
    [0.0, 1.0, 0.0, 0.5, 1.0],   # MAS
    [0.0, 1.0, 1.0, 0.5, 0.0],   # LTO
    [0.0, 0.0, 0.0, 0.0, 0.5],   # IDV
    [0.0, 1.0, 0.0, 0.0, 0.0],   # UAI
])

fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=200)
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(checks))); ax.set_xticklabels(checks, fontsize=12)
ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims, fontsize=12)
for i in range(len(dims)):
    for j in range(len(checks)):
        v = M[i, j]
        lab = "pass" if v == 1 else ("partial" if v == 0.5 else "fail")
        ax.text(j, i, lab, ha="center", va="center", fontsize=11.5,
                color="#111", fontweight="bold" if v == 1 else "normal")
ax.set_title("Four-fold robustness evidence matrix", fontsize=15.5, pad=14)
ax.set_xlabel("Independent checks", fontsize=13)
ax.set_ylabel("Hofstede dimension", fontsize=13)
# 核心结论标注
ax.add_patch(plt.Rectangle((-0.5, -0.5), 5, 1, fill=False, edgecolor="#1a1a1a", lw=2.6))
ax.text(5.1, 0, "core finding\n(survives all)", va="center", fontsize=11.5, color="#1a1a1a")
fig.tight_layout()
fig.savefig(FIGS / "figB1_robustness.png", bbox_inches="tight")
plt.close(fig)
print("-> figB1_robustness.png")

# ---------- figB2 构念边界 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=200)
# 左：Hofstede 六维秩相关
d6 = ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]
rho6 = [-0.50, 0.50, 0.00, -0.50, 0.00, -1.00]
cols = ["#C0392B" if r <= 0 else "#2E7D5B" for r in rho6]
ax1.bar(d6, rho6, color=cols, alpha=0.9)
ax1.axhline(0, color="#333", lw=1)
ax1.set_ylim(-1.15, 1.15)
ax1.set_ylabel("Spearman's rho (text vs. survey)", fontsize=12.5)
ax1.set_title("Hofstede dimensions: text–survey convergence", fontsize=14, pad=10)
ax1.grid(axis="y", alpha=0.3)
for i, r in enumerate(rho6):
    ax1.text(i, r + (0.06 if r >= 0 else -0.13), f"{r:+.2f}", ha="center", fontsize=12)
# 右：WVS 13 刻面方向一致率
ax2.bar(["agree", "disagree"], [7, 6], color=["#2E7D5B", "#C0392B"], alpha=0.9, width=0.55)
ax2.set_ylabel("Number of WVS facets", fontsize=12.5)
ax2.set_title("WVS 13 facets: direction agreement with survey means", fontsize=14, pad=10)
ax2.grid(axis="y", alpha=0.3)
ax2.text(0, 7.15, "7 / 13\n(chance level)", ha="center", fontsize=12)
ax2.set_ylim(0, 9)
fig.tight_layout()
fig.savefig(FIGS / "figB2_construct_boundary.png", bbox_inches="tight")
plt.close(fig)
print("-> figB2_construct_boundary.png")
