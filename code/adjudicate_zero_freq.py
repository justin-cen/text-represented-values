# -*- coding: utf-8 -*-
"""51 个零频锚定词的机器裁定（两级）。

一级：全语料频度复检（Aho-Corasick，全量语料而非抽样）。
二级：留一法影响检验（对真零频词，从刻面质心剔除后看对比分变化）。

输出: work/lexicon/frozen_v3/adjudication_51.csv
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REV = BASE / "work" / "lexicon" / "frozen_v3" / "lexicon_v3_review.csv"
FR = BASE / "work" / "lexicon" / "frozen_v3" / "lexicon_v3_frozen.json"

rev = pd.read_csv(REV)
zero = rev[(rev.freq_total == 0) & (rev.source == "anchor")].copy()
print(f"抽样零频锚定词: {len(zero)} 个")

# ---------- 一级：全语料频度 ----------
import ahocorasick
CORPORA = {
    "douban": ("reviews_zh/reviews_zh.parquet", "zh"),
    "letterboxd": ("reviews_en/reviews_en_letterboxd.parquet", "en"),
    "news_zh": ("news_zh_recent/news_zh_recent.parquet", "zh"),
    "news_en": ("news_en_recent/news_en_recent.parquet", "en"),
    "thucnews": ("news_zh/news_zh.parquet", "zh"),
    "xsum": ("news_en/news_en.parquet", "en"),
}
terms_zh = zero.loc[zero.lang == "zh", "word"].tolist()
terms_en = zero.loc[zero.lang == "en", "word"].str.lower().tolist()
A = ahocorasick.Automaton()
for t in terms_zh:
    A.add_word(t, t)
for t in terms_en:
    A.add_word(" " + t + " ", t)
A.make_automaton()
freq = {t: {} for t in terms_zh + terms_en}
for name, (rel, lang) in CORPORA.items():
    p = BASE / "data_corpus" / rel
    if not p.exists():
        continue
    df = pd.read_parquet(p, columns=["text"])
    n = len(df)
    print(f"  扫描 {name}: {n:,} 篇…", flush=True)
    for text in df["text"]:
        tl = (" " + text + " ") if lang == "zh" else (" " + text.lower() + " ")
        for _, t in A.iter(tl):
            freq[t][name] = freq[t].get(name, 0) + 1
zero["full_freq"] = zero["word"].map(
    lambda w: sum(freq.get(w, {}).values()) if w in freq else
    sum(freq.get(str(w).lower(), {}).values()))
zero["full_freq"] = zero.apply(
    lambda r: sum(freq.get(r["word"] if r["lang"] == "zh" else str(r["word"]).lower(), {}).values()),
    axis=1)
print("\n一级频度分布:")
print(zero.groupby(pd.cut(zero.full_freq, [-1, 0, 9, 10**9],
                                          labels=["真零频", "1-9次", "≥10次"])).size().to_string())

zero.to_csv(BASE / "work" / "lexicon" / "frozen_v3" / "adjudication_51_stage1.csv",
            index=False, encoding="utf-8-sig")
print("-> adjudication_51_stage1.csv")
