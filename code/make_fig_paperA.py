# -*- coding: utf-8 -*-
"""论文A 补充图：
figA2 时间稳定性（豆瓣十年 + 人民日报逐月）
figA3 中西价值观画像对照（4 家中文官媒 vs 12 家西方媒体）
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

SCV = ["富强", "民主", "文明", "和谐", "自由", "平等", "公正", "法治", "爱国", "敬业", "诚信", "友善"]
x = pd.ExcelFile(BASE / "scv_china_side.xlsx")
db = x.parse("douban_yearly_trend")
rm = x.parse("rm_monthly_trend")
zhm = x.parse("zh_media_profile")
wem = x.parse("west_media_profile")

# ---------- figA2 时间稳定性 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.6), dpi=200)
show = ["富强", "敬业", "文明", "和谐"]
cols = ["#8C1D18", "#C0392B", "#2E7D5B", "#2E5E8C"]
for c, col in zip(show, cols):
    ax1.plot(db["year"], db[c], marker="o", ms=5, lw=2, color=col, label=c)
    ax2.plot(rm["month"], rm[c], marker="s", ms=5, lw=2, color=col, label=c)
ax1.set_xlabel("年份", fontsize=13); ax1.set_ylabel("相对凸显度（ipsative ×1000）", fontsize=13)
ax1.set_title("豆瓣影评的十年价值观轨迹（2010—2019）", fontsize=15, pad=10)
ax1.legend(frameon=False, fontsize=11.5, ncol=2); ax1.grid(alpha=0.3)
ax1.set_xticks(db["year"].values)

ax2.set_xlabel("月份（2026 年）", fontsize=13)
ax2.set_title("人民日报的逐月价值观轨迹（9 个月）", fontsize=15, pad=10)
ax2.legend(frameon=False, fontsize=11.5, ncol=2); ax2.grid(alpha=0.3)
ax2.set_xticks(rm["month"].values)
# 波动幅度标注
for ax, d, key in [(ax1, db, "year"), (ax2, rm, "month")]:
    rng = d["富强"].max() - d["富强"].min()
    ax.annotate(f"富强波动幅度仅 {rng:.1f}", xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=11.5, color="#8C1D18")
fig.tight_layout()
fig.savefig(FIGS / "figA2_stability.png", bbox_inches="tight")
plt.close(fig)
print("-> figA2_stability.png")

# ---------- figA3 中西对照 ----------
fig, ax = plt.subplots(figsize=(15, 6.2), dpi=200)
zz = zhm[SCV].mean(axis=0)
zw = wem[SCV].mean(axis=0)
xx = np.arange(len(SCV)); w = 0.36
ax.bar(xx - w / 2, zz.values, w, label="中文官方媒体（4 家均值）", color="#C0392B", alpha=0.9)
ax.bar(xx + w / 2, zw.values, w, label="西方媒体（12 家均值）", color="#2E5E8C", alpha=0.9)
ax.axhline(0, color="#444", lw=0.9)
ax.set_xticks(xx); ax.set_xticklabels(SCV, fontsize=13)
ax.set_ylabel("相对凸显度（ipsative ×1000）", fontsize=13)
ax.set_title("中西媒体话语的社会主义核心价值观画像对照", fontsize=15.5, pad=12)
ax.legend(frameon=False, fontsize=12.5); ax.grid(axis="y", alpha=0.3)
# 标注差异最大的三项
diff = (zz - zw).abs().sort_values(ascending=False).head(3)
for k in diff.index:
    i = SCV.index(k)
    ax.annotate("", xy=(i, max(zz[k], zw[k]) + 2.2), xytext=(i, max(zz[k], zw[k]) + 0.6),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.8))
    ax.text(i, max(zz[k], zw[k]) + 2.6, f"Δ={zz[k]-zw[k]:+.1f}", ha="center", fontsize=11.5, color="#333")
fig.tight_layout()
fig.savefig(FIGS / "figA3_china_west.png", bbox_inches="tight")
plt.close(fig)
print("-> figA3_china_west.png")

print("\n中文官媒均值 Top4:", zz.sort_values(ascending=False).head(4).round(1).to_dict())
print("西方媒体均值 Top4:", zw.sort_values(ascending=False).head(4).round(1).to_dict())
print("差异最大三项:", (zz - zw).abs().sort_values(ascending=False).head(3).round(2).to_dict())
