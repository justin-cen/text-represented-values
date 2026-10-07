# -*- coding: utf-8 -*-
"""构建同年对齐新闻集：人民日报 2025（zh）× FineNews 2025（west），按月等量配对。

严格同年（2025×2025）对齐，消除 ±1 年错位。
用法：
  python build_news_sameyear.py                 # 用扩充语料 rmrb_2025_full.parquet（若有）
  python build_news_sameyear.py --per-month 1500  # 指定每月每侧篇数（默认自动取两侧较小值）
输出 analysis_sets/news_sameyear_2025.parquet
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--per-month", type=int, default=None, help="每月每侧上限（默认取两侧较小值）")
ap.add_argument("--tag", default="_full", help="中文侧语料后缀（_full 为扩充版，空为初版）")
args = ap.parse_args()

# 中文侧：人民日报 2025（优先扩充版）
zh_path = BASE / "raw" / "scrape_2026" / f"rmrb_2025{args.tag}.parquet"
if not zh_path.exists():
    zh_path = BASE / "raw" / "scrape_2026" / "rmrb_2025.parquet"
print("中文侧语料:", zh_path.name)
zh = pd.read_parquet(zh_path)
zh["match_month"] = zh["date"].str[5:7]
zh["text"] = zh["text"].astype(str)
zh = zh[zh["text"].str.len() >= 200].copy()
zh["year"] = 2025

# 西方侧：复用 news_aligned_month 的 FineNews 2025，按月抽样
west_all = pd.read_parquet(BASE / "analysis_sets" / "news_aligned_month.parquet")
west_all = west_all[west_all["culture"] == "west"].copy()
west_all["match_month"] = west_all["match_month"].astype(str).str.zfill(2)

n_by_month = zh.groupby("match_month").size().to_dict()
west_parts = []
tot_zh = tot_west = 0
for mm, n in sorted(n_by_month.items()):
    pool = west_all[west_all["match_month"] == mm]
    take = min(n, len(pool))
    if args.per_month:
        take = min(take, args.per_month)
    if take == 0:
        continue
    west_parts.append(pool.sample(n=take, random_state=42))
    tot_zh += take
    tot_west += take
west = pd.concat(west_parts, ignore_index=True)
west["year"] = 2025

# 中文侧按同样配额裁剪，保证按月等量
zh_parts = []
for mm, n in sorted(n_by_month.items()):
    pool = zh[zh["match_month"] == mm]
    take = min(n, len(west[west["match_month"] == mm]))
    if take == 0:
        continue
    zh_parts.append(pool.sample(n=take, random_state=42))
zh = pd.concat(zh_parts, ignore_index=True)
print(f"按月等量配对完成：中文 {len(zh):,} × 西方 {len(west):,}")

cols = ["corpus_id", "culture", "genre", "source", "channel", "item", "date", "year",
        "rating", "sentiment", "title", "text", "meta", "match_month"]
out = pd.concat([zh[cols], west[cols]], ignore_index=True)
out.to_parquet(BASE / "analysis_sets" / "news_sameyear_2025.parquet", index=False)

print("同年对齐集:", len(out))
print(out.groupby(["culture", "match_month"]).size().unstack(0).to_string())
print("中文侧来源:", out[out.culture == "zh"]["source"].value_counts().to_dict())
print("西方侧来源:", out[out.culture == "west"]["source"].value_counts().to_dict())
print("-> analysis_sets/news_sameyear_2025.parquet")
