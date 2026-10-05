# -*- coding: utf-8 -*-
"""构建同年对齐新闻集：人民日报 2025（zh）× FineNews 2025（west），按月等量配对。

解决"相邻年份（±1 年）对齐"局限。输出 analysis_sets/news_sameyear_2025.parquet
"""
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 中文侧：新抓的人民日报 2025
zh = pd.read_parquet(BASE / "raw" / "scrape_2026" / "rmrb_2025.parquet")
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
for mm, n in sorted(n_by_month.items()):
    pool = west_all[west_all["match_month"] == mm]
    take = min(n, len(pool))
    west_parts.append(pool.sample(n=take, random_state=42))
west = pd.concat(west_parts, ignore_index=True)
west["year"] = 2025

cols = ["corpus_id", "culture", "genre", "source", "channel", "item", "date", "year",
        "rating", "sentiment", "title", "text", "meta", "match_month"]
out = pd.concat([zh[cols], west[cols]], ignore_index=True)
out.to_parquet(BASE / "analysis_sets" / "news_sameyear_2025.parquet", index=False)

print("同年对齐集:", len(out))
print(out.groupby(["culture", "match_month"]).size().unstack(0).to_string())
print("中文侧来源:", out[out.culture == "zh"]["source"].value_counts().to_dict())
print("西方侧来源:", out[out.culture == "west"]["source"].value_counts().to_dict())
print("-> analysis_sets/news_sameyear_2025.parquet")
