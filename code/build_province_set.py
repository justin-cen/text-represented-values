# -*- coding: utf-8 -*-
"""把省级新闻语料转为 analysis_set 格式（复用嵌入管线打分）。"""
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

src = BASE / "raw/province_news/pd_news.parquet"
d = pd.read_parquet(src)
print(f"源: {len(d)} 篇 / {d.province.nunique()} 省")

out = pd.DataFrame({
    "corpus_id": d["corpus_id"],
    "culture": "zh",
    "genre": "news",
    "source": "people_province",
    "channel": d["province"],          # 省级归属
    "item": d["title"],
    "date": d["date"],
    "year": d["year"],
    "rating": None,
    "sentiment": None,
    "title": d["title"],
    "text": d["text"],
    "meta": d["meta"],
})
# 过滤过短文本
out = out[out.text.str.len() >= 200].copy()
out.to_parquet(BASE / "analysis_sets/province_news.parquet", index=False)
print(f"-> analysis_sets/province_news.parquet: {len(out)} 篇 / {out.channel.nunique()} 省")
print(out.groupby("channel").size().sort_values(ascending=False).to_string())
