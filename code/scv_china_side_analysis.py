# -*- coding: utf-8 -*-
"""中文侧典型价值观影响分析（SCV 聚焦小节数据与图）。

产出:
  results/scv_china_side.csv          分媒体/分体裁 SCV 画像表
  work/paper/figs/fig6_scv_china.png  中文侧典型价值观画像+时间轨迹图
"""
import io
import json
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

SCV_ORDER = ["SCV_prosperity", "SCV_democracy", "SCV_civility", "SCV_harmony",
             "SCV_freedom", "SCV_equality", "SCV_justice", "SCV_ruleoflaw",
             "SCV_patriotism", "SCV_dedication", "SCV_integrity", "SCV_friendliness"]
SCV_CN = {"SCV_prosperity": "富强", "SCV_democracy": "民主", "SCV_civility": "文明",
          "SCV_harmony": "和谐", "SCV_freedom": "自由", "SCV_equality": "平等",
          "SCV_justice": "公正", "SCV_ruleoflaw": "法治", "SCV_patriotism": "爱国",
          "SCV_dedication": "敬业", "SCV_integrity": "诚信", "SCV_friendliness": "友善"}
KEY4 = ["SCV_prosperity", "SCV_democracy", "SCV_dedication", "SCV_harmony"]


def ips(df):
    d = df[df.system == "scv"].copy()
    d["ips"] = d["sim"] - d.groupby("corpus_id")["sim"].transform("mean")
    return d


def main():
    news = pd.read_parquet(BASE / "results" / "news_aligned_month_facet_sims_v3.parquet")
    films = pd.read_parquet(BASE / "results" / "reviews_samefilms_v3_facet_sims_v3.parquet")

    # ---------- 1) 分媒体 SCV 画像 ----------
    n = ips(news)
    chan_prof = n.groupby(["channel", "facet"])["ips"].mean().unstack("facet").reindex(columns=SCV_ORDER)
    chan_prof["culture"] = n.groupby("channel")["culture"].first()
    zh_chan = chan_prof[chan_prof.culture == "zh"].drop(columns="culture") * 1000
    we_chan = chan_prof[chan_prof.culture == "west"].drop(columns="culture") * 1000
    print("=== 中文媒体 SCV 画像（ipsative ×1000）===")
    print(zh_chan.round(1).rename(columns=SCV_CN).to_string())

    # ---------- 2) 中文侧时间轨迹（人民日报按月） ----------
    rm = n[n["channel"] == "人民日报"].copy()
    rm["month"] = rm["date"].str[5:7]
    trend = rm.groupby(["month", "facet"])["ips"].mean().unstack("facet").reindex(columns=SCV_ORDER) * 1000
    print("\n=== 人民日报 SCV 月度轨迹（节选）===")
    print(trend[KEY4].round(2).rename(columns=SCV_CN).to_string())

    # ---------- 3) 豆瓣影评按年份的 SCV 轨迹 ----------
    f = ips(films)
    fz = f[f.culture == "zh"].copy()
    fz["year"] = pd.to_datetime(fz["date"], errors="coerce").dt.year
    yearly = fz.groupby(["year", "facet"])["ips"].mean().unstack("facet").reindex(columns=SCV_ORDER) * 1000
    yearly = yearly[yearly.index >= 2010]
    print("\n=== 豆瓣影评 SCV 年度轨迹（节选，≥2010）===")
    print(yearly[KEY4].round(2).rename(columns=SCV_CN).to_string())

    # ---------- 4) 汇总 CSV ----------
    out = {
        "zh_media_profile": zh_chan,
        "west_media_profile": we_chan,
        "rm_monthly_trend": trend,
        "douban_yearly_trend": yearly,
    }
    with pd.ExcelWriter(BASE / "results" / "scv_china_side.xlsx") as w:
        for k, v in out.items():
            v.rename(columns=SCV_CN).to_excel(w, sheet_name=k)
    print("\n-> results/scv_china_side.xlsx")

    # ---------- 5) 图6 ----------
    fig = plt.figure(figsize=(12.5, 4.6), dpi=200)
    # 左：中文媒体 SCV 画像（人民日报 vs 新华网/中新网 vs 西方均值）
    ax1 = fig.add_subplot(121)
    x = np.arange(len(SCV_ORDER))
    ax1.bar(x - 0.25, zh_chan.loc["人民日报", SCV_ORDER], width=0.25,
            color="#C0392B", label="人民日报")
    ax1.bar(x, zh_chan.loc["新华网", SCV_ORDER], width=0.25,
            color="#E67E22", label="新华网")
    ax1.bar(x + 0.25, we_chan.mean().reindex(SCV_ORDER), width=0.25,
            color="#2E5E8C", label="西方媒体均值")
    ax1.axhline(0, color="#333", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCV_CN[k] for k in SCV_ORDER], fontsize=9, rotation=45)
    ax1.set_ylabel("ipsative 凸显度（×1000）")
    ax1.set_title("图6a 社会主义核心价值观画像：中文媒体 vs 西方媒体", fontsize=10.5)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.3)
    # 右：人民日报月度轨迹（四个代表值）
    ax2 = fig.add_subplot(122)
    for k, c in zip(KEY4, ["#C0392B", "#2E5E8C", "#27AE60", "#8E44AD"]):
        ax2.plot(trend.index, trend[k], marker="o", lw=1.8, color=c, label=SCV_CN[k])
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xlabel("2026 年月份")
    ax2.set_ylabel("ipsative 凸显度（×1000）")
    ax2.set_title("图6b 人民日报典型价值观月度轨迹（2026-01~09）", fontsize=10.5)
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_scv_china.png", bbox_inches="tight")
    print("fig6 done")


if __name__ == "__main__":
    main()
