# -*- coding: utf-8 -*-
"""跨年代对照与分媒体稳健性分析（等历史层嵌入完成后运行）。

分析一（跨年代）：同一体系的维度对比分在三个年代层的取值
  - 历史层新闻：THUCNews(2005-2011, zh) vs XSum/BBC(2010-2017, west)
  - 近期层新闻：人民日报系(2026) vs FineNews(2025)
  - 历史层影评：豆瓣(≤2019) vs IMDB(≤2011)
  - 近期层影评：豆瓣(2019抓取) vs Letterboxd(2025抓取)
分析二（分媒体稳健性）：新闻侧按 outlet 拆分的对比分矩阵（人民日报 vs 新华/中新/央广；UK vs US）

用法: python era_outlet_analysis.py [--suffix _v3]
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PAIRS = {"IDV": ("IDV_col", "IDV_ind"), "PDI": ("PDI_high", "PDI_low"),
         "MAS": ("MAS_mas", "MAS_fem"), "UAI": ("UAI_high", "UAI_low"),
         "LTO": ("LTO_long", "LTO_short"), "IND": ("IND_res", "IND_ind")}

ERA_SETS = [
    ("新闻·历史层", "news_general", "THUCNews05-11", "XSum/BBC10-17"),
    ("新闻·近期层", "news_aligned_month", "人民日报系2026", "FineNews2025"),
    ("新闻·近期窗口", "news_recent_2026", "中文2026-08~09", "西方2025-08~10"),
    ("影评·历史层", "reviews_general", "豆瓣", "IMDB"),
    ("影评·历史豆瓣×IMDB", "reviews_douban_imdb_matched_sample", "豆瓣≤2011", "IMDB 2011(tt直连)"),
    ("影评·历史豆瓣×genome", "reviews_douban_genome_matched_sample", "豆瓣≤2012", "IMDB 2012±(tt直连)"),
    ("影评·西片对照", "reviews_westernfilms", "豆瓣西片", "IMDB"),
    ("影评·同片配对", "reviews_samefilms_v3", "豆瓣(年度校验)", "Letterboxd"),
    ("影评·同片同时代", "reviews_samefilms_recent_v3", "豆瓣2020-26抓取", "Letterboxd2025"),
]


def contrasts(setname: str, suffix: str) -> dict | None:
    p = BASE / "results" / f"{setname}_facet_sims{suffix}.parquet"
    if not p.exists():
        return None
    sims = pd.read_parquet(p)
    h = sims[sims.system == "hofstede"]
    w = h.pivot_table(index=["corpus_id", "culture"], columns="facet", values="sim")
    out = {}
    for dim, (a, b) in PAIRS.items():
        try:
            c = (w[a] - w[b]).groupby("culture").mean()
        except KeyError:
            return None
        out[dim] = (c.get("zh", np.nan) - c.get("west", np.nan)) * 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    args = ap.parse_args()

    print("=" * 78)
    print("分析一：维度对比分（zh−west, ×1000）的跨年代/体裁对照")
    print("=" * 78)
    rows = []
    for label, setname, zh_name, we_name in ERA_SETS:
        c = contrasts(setname, args.suffix)
        if c is None:
            print(f"[跳过] {label} ({setname}) —— 结果文件尚未生成")
            continue
        rows.append({"层": label, "中文侧": zh_name, "西方侧": we_name,
                     **{k: round(v, 2) for k, v in c.items()}})
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        out = BASE / "results" / f"era_contrasts{args.suffix}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n-> {out}")
        # 方向一致性矩阵
        print("\n方向一致性（同一维度跨层符号是否一致）:")
        dims = list(PAIRS)
        for d in dims:
            signs = [np.sign(r.get(d, np.nan)) for r in rows]
            n_pos = sum(1 for s in signs if s > 0)
            n_neg = sum(1 for s in signs if s < 0)
            flag = "★ 完全一致" if max(n_pos, n_neg) == len(signs) else f"{max(n_pos, n_neg)}/{len(signs)} 一致"
            print(f"  {d:4s} {flag}  (正{int(n_pos)} 负{int(n_neg)})")

    print("\n" + "=" * 78)
    print("分析二：新闻分媒体稳健性（outlet 级对比分, ×1000）")
    print("=" * 78)
    for setname in ["news_aligned_month", "news_general"]:
        p = BASE / "results" / f"{setname}_facet_sims{args.suffix}.parquet"
        if not p.exists():
            continue
        sims = pd.read_parquet(p)
        h = sims[(sims.system == "hofstede") & (sims.channel.notna())]
        w = h.pivot_table(index=["channel", "corpus_id"], columns="facet", values="sim")
        tab = {}
        for dim, (a, b) in PAIRS.items():
            tab[dim] = ((w[a] - w[b]).groupby("channel").mean() * 1000).round(1)
        T = pd.DataFrame(tab)
        T["n"] = h.groupby("channel")["corpus_id"].nunique()
        T = T.sort_values("n", ascending=False)
        print(f"\n--- {setname} ---")
        print(T.to_string())
        T.to_csv(BASE / "results" / f"{setname}_outlet_contrasts{args.suffix}.csv",
                 encoding="utf-8-sig")


if __name__ == "__main__":
    main()
