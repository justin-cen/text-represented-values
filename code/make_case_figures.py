# -*- coding: utf-8 -*-
"""典型案例雷达图：选文化差异最极端/最一致的影片与媒体，绘制价值观画像雷达图。

产出 work/paper/figs/fig4_case_films.png（影片）、fig5_case_outlets.png（媒体）。
用法: python make_case_figures.py [--set reviews_samefilms_v3] [--topn 8]
"""
import argparse
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
FIGS.mkdir(exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

POLES = ["PDI_high", "PDI_low", "IDV_ind", "IDV_col", "MAS_mas", "MAS_fem",
         "UAI_high", "UAI_low", "LTO_long", "LTO_short", "IND_ind", "IND_res"]
POLE_CN = ["高权力距离", "低权力距离", "个人主义", "集体主义", "男性气质", "女性气质",
           "高不确定回避", "低不确定回避", "长期导向", "短期导向", "放纵", "约束"]


def radar(ax, zh, we, title):
    n = len(zh)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    zh2 = list(zh) + [zh[0]]
    we2 = list(we) + [we[0]]
    ang = angles + [angles[0]]
    ax.plot(ang, zh2, color="#C0392B", lw=2, label="中文侧")
    ax.fill(ang, zh2, color="#C0392B", alpha=0.15)
    ax.plot(ang, we2, color="#2E5E8C", lw=2, ls="--", label="西方侧")
    ax.fill(ang, we2, color="#2E5E8C", alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(POLE_CN, fontsize=11)
    ax.set_title(title, fontsize=13, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.14), fontsize=10, frameon=False)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(alpha=0.35)


def profile(sims, mask_col, mask_val):
    sub = sims[(sims.system == "hofstede") & (sims[mask_col] == mask_val)]
    sub = sub.copy()
    sub["ips"] = sub["sim"] - sub.groupby("corpus_id")["sim"].transform("mean")
    t = sub.groupby(["facet", "culture"])["ips"].mean().unstack()
    t = t.reindex(POLES)
    return t["zh"].values * 1000, t["west"].values * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="reviews_samefilms_v2")
    ap.add_argument("--news-set", default="news_aligned_month")
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args()

    # ---------------- 影片案例 ----------------
    sims = pd.read_parquet(BASE / "results" / f"{args.set}_facet_sims_v3.parquet")
    h = sims[sims.system == "hofstede"].copy()
    h["ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
    wide = h.groupby(["matched_film", "facet", "culture"])["ips"].mean() \
            .unstack(["facet", "culture"])          # index=film, columns=(facet,culture)
    zh_mat = wide.xs("zh", axis=1, level="culture").reindex(columns=POLES)
    we_mat = wide.xs("west", axis=1, level="culture").reindex(columns=POLES)
    div = (zh_mat - we_mat).abs().sum(axis=1).dropna().sort_values(ascending=False)
    nside = h.groupby(["matched_film", "culture"]).size().unstack()
    ok = nside[(nside["zh"] >= 3) & (nside["west"] >= 3)].index
    div = div.loc[div.index.intersection(ok)]
    top_div = div.head(args.topn)
    top_con = div.tail(args.topn)
    print("差异最大影片 TOP:")
    for f, v in top_div.head(5).items():
        print(f"  {f}: L1距离={v:.3f}")
    print("最一致影片 TOP:")
    for f, v in top_con.tail(3).items():
        print(f"  {f}: L1距离={v:.3f}")

    picks = [top_div.index[0], top_div.index[min(1, len(top_div) - 1)],
             top_div.index[min(2, len(top_div) - 1)], top_con.index[-1]]
    labels = ["差异最大 #1", "差异最大 #2", "差异最大 #3", "最一致"]
    films_meta = pd.read_csv(BASE / "analysis_sets" / f"{args.set}_films.csv") \
        if (BASE / "analysis_sets" / f"{args.set}_films.csv").exists() else None

    fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(10.5, 10.5), dpi=200)
    for ax, pk, lb in zip(axes.flat, picks, labels):
        zh = zh_mat.loc[pk].values * 1000
        we = we_mat.loc[pk].values * 1000
        name = pk
        if films_meta is not None and "douban_name" in films_meta.columns:
            row = films_meta[films_meta["matched_film"] == pk]
            if len(row):
                name = row.iloc[0]["douban_name"]
        radar(ax, zh, we, f"{lb}：{name}")
    fig.suptitle("典型影片的中西价值观画像雷达图（Hofstede 12 极, ipsative ×1000）",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_case_films.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")

    # ---------------- 媒体案例 ----------------
    nsims = pd.read_parquet(BASE / "results" / f"{args.news_set}_facet_sims_v3.parquet")
    nh = nsims[nsims.system == "hofstede"].copy()
    nh["ips"] = nh["sim"] - nh.groupby("corpus_id")["sim"].transform("mean")
    prof = nh.groupby(["channel", "facet"])["ips"].mean().unstack("facet").reindex(columns=POLES)
    prof["culture"] = nh.groupby("channel")["culture"].first()
    zh_channels = prof[prof.culture == "zh"].drop(columns="culture")
    we_channels = prof[prof.culture == "west"].drop(columns="culture")
    zh_main = "人民日报"
    base = zh_channels.loc[zh_main]
    divc = (we_channels - base).abs().sum(axis=1).sort_values(ascending=False)
    print("\n与人民日报差异最大的西方媒体 TOP:")
    for f, v in divc.head(6).items():
        print(f"  {f}: L1距离={v:.3f}")
    west_most_div = divc.index[0]
    west_least_div = divc.index[-1]
    picks2 = [(zh_main, west_most_div, f"人民日报 vs {west_most_div}（差异最大）"),
              (zh_main, "Independent", "人民日报 vs Independent"),
              (zh_main, "Fox", "人民日报 vs Fox"),
              (zh_main, west_least_div, f"人民日报 vs {west_least_div}（最接近）")]
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(10.5, 10.5), dpi=200)
    for ax, (z, wch, lb) in zip(axes.flat, picks2):
        zh = zh_channels.loc[z].values * 1000
        we = we_channels.loc[wch].values * 1000
        radar(ax, zh, we, lb)
    fig.suptitle("典型媒体的中西价值观画像雷达图（Hofstede 12 极, ipsative ×1000）",
                 fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_case_outlets.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")
    print("ALL ->", FIGS)


if __name__ == "__main__":
    main()
