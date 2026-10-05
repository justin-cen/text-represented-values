# -*- coding: utf-8 -*-
"""将 genome_2021 的 reviews.json（262万行 jsonl）整理为带 tt 的英文影评语料。

链: reviews.item_id → metadata.imdbId → tt（补前缀）
输出: data_corpus/reviews_en/reviews_en_genome.parquet
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
RAW = BASE / "raw" / "genome_2021" / "movie_dataset_public_final" / "raw"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# metadata: item_id -> imdbId, title
meta = {}
with open(RAW / "metadata.json", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        iid = d.get("imdbId")
        if iid:
            meta[d["item_id"]] = {"imdbId": str(iid), "title": d.get("title", ""),
                                  "avgRating": d.get("avgRating")}
print(f"metadata 影片 {len(meta):,}")

rows = []
with open(RAW / "reviews.json", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i % 500000 == 0 and i:
            print(f"  已读 {i:,} 行…", flush=True)
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = meta.get(d.get("item_id"))
        if not m or not m.get("imdbId"):
            continue
        t = " ".join((d.get("txt") or "").split())
        if len(t) < 50:
            continue
        tt = m["imdbId"]
        if not tt.startswith("tt"):
            tt = "tt" + tt
        rows.append({"tt": tt, "title": m["title"], "text": t,
                     "avgRating_ml": m.get("avgRating"), "item_id": d["item_id"]})

df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
import hashlib
df["corpus_id"] = ["gm_" + hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
                   for t in df["text"]]
out = BASE / "reviews_en" / "reviews_en_genome.parquet"
df.to_parquet(out, index=False)
print(f"-> {out}  rows={len(df):,}  影片数={df['tt'].nunique():,}")
