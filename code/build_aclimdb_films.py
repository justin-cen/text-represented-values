# -*- coding: utf-8 -*-
"""将 aclImdb 原始数据整理为带 tt 影片编号的英文影评语料。

urls_*.txt 第 i 行 = 第 i 条评论所在影片的 IMDB 页面（含 tt 编号）。
文件名 {i}_{rating}.txt，rating 1-10（train pos/neg）；unsup 无评分（label=-1）。

输出: data_corpus/reviews_en/reviews_en_imdb_films.parquet
"""
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
SRC = BASE / "raw" / "aclImdb" / "aclImdb"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TT = re.compile(r"/title/(tt\d+)/")

rows = []
for split in ["train", "test"]:
    for lab in ["pos", "neg"]:
        ufile = SRC / split / f"urls_{lab}.txt"
        if not ufile.exists():
            continue
        urls = ufile.read_text(encoding="utf-8", errors="ignore").splitlines()
        label = "pos" if lab == "pos" else "neg"
        folder = SRC / split / lab
        for i, u in enumerate(urls):
            m = TT.search(u)
            if not m:
                continue
            f = folder / f"{i}_"
            # 文件名是 {i}_{rating}.txt
            cands = list(folder.glob(f"{i}_*.txt"))
            if not cands:
                continue
            text = cands[0].read_text(encoding="utf-8", errors="ignore")
            rating = int(cands[0].stem.split("_")[1]) if "_" in cands[0].stem else None
            rows.append({"tt": m.group(1), "rating": rating, "sentiment": label,
                         "text": " ".join(text.split()), "split": split})
    # unsup
    ufile = SRC / split / "urls_unsup.txt"
    if ufile.exists():
        urls = ufile.read_text(encoding="utf-8", errors="ignore").splitlines()
        folder = SRC / split / "unsup"
        for i, u in enumerate(urls):
            m = TT.search(u)
            if not m:
                continue
            cands = list(folder.glob(f"{i}_*.txt"))
            if not cands:
                continue
            text = cands[0].read_text(encoding="utf-8", errors="ignore")
            rows.append({"tt": m.group(1), "rating": None, "sentiment": None,
                         "text": " ".join(text.split()), "split": split})

df = pd.DataFrame(rows)
df = df[df["text"].str.len() >= 50].drop_duplicates(subset=["text"])
import hashlib
df["corpus_id"] = ["imt_" + hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
                   for t in df["text"]]
out = BASE / "reviews_en" / "reviews_en_imdb_films.parquet"
df.to_parquet(out, index=False)
print(f"-> {out}  rows={len(df):,}  影片数={df['tt'].nunique():,}")
print("sentiment:", df["sentiment"].value_counts(dropna=False).to_dict())
print("split:", df["split"].value_counts().to_dict())
print("年份跨度: 待与 title.basics 关联")
