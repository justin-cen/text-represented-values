# -*- coding: utf-8 -*-
"""合并四个政治话语 JSONL → reports_political.parquet（统一 13 列 + 保留 year/country/title/url/n_chars）。

用法: python build_political_corpus.py
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
SRC = BASE / "raw" / "political"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FILES = ["gwr_zh.jsonl", "sotu_us.jsonl", "kings_speech_uk.jsonl", "soteu_eu.jsonl"]

rows = []
for fn in FILES:
    p = SRC / fn
    if not p.exists():
        print(f"!! 缺 {fn}")
        continue
    n = 0
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows.append({
            "corpus_id": d["corpus_id"], "culture": d["culture"], "genre": "political_report",
            "source": d["source"], "channel": d["channel"], "item": d["title"],
            "date": d.get("date"), "year": d.get("year"), "rating": None, "sentiment": None,
            "title": d.get("title"), "text": d["text"],
            "meta": json.dumps({"country": d.get("country"), "url": d.get("url"),
                                "n_chars": d.get("n_chars")}, ensure_ascii=False),
            "matched_film": None,
        })
        n += 1
    print(f"{fn}: {n} 份")

df = pd.DataFrame(rows)
# 防重复（同 corpus_id 只保留第一份）
before = len(df)
df = df.drop_duplicates(subset=["corpus_id"], keep="first").reset_index(drop=True)
if len(df) < before:
    print(f"去重: {before} -> {len(df)}")
out = BASE / "analysis_sets" / "reports_political.parquet"
df.to_parquet(out, index=False)
print(f"\n-> {out}  rows={len(df):,}")
print("按 channel 分布:", df.groupby(["channel", "culture"]).size().to_dict())
print("按 year 范围:", df.groupby("channel")["year"].agg(["min", "max"]).to_dict())
print("文本长度: mean={:.0f} p50={:.0f} max={:.0f}".format(
    df["text"].str.len().mean(), df["text"].str.len().median(), df["text"].str.len().max()))
