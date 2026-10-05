# -*- coding: utf-8 -*-
"""截断稳健性检验（批量高效版）：比较"前 2000 字符截断"与"全文分块均值"的对比分方向。

关键：把所有文本的所有块拼成一个列表，一次性批量编码，再按文本聚合（避免逐篇调用）。
输出 results/truncation_check.csv
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
CORPUS = BASE / "data_corpus"
LEX = BASE / "work" / "lexicon"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
POLES = [p for _, a, b in DIMS for p in (a, b)]

ap = argparse.ArgumentParser()
ap.add_argument("--per-set", type=int, default=1500)
ap.add_argument("--chunk", type=int, default=2000)
args = ap.parse_args()

fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")


def emb(texts, prefix):
    return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=96, show_progress_bar=False)


cent = {}
for p in POLES:
    ws = [x for x in (fr["hofstede"][p]["en"] + fr["hofstede"][p]["zh"]) if x and len(x) < 40]
    E = emb(ws, "query: ")
    c = E.mean(axis=0)
    cent[p] = c / np.linalg.norm(c)
CM = np.stack([cent[p] for p in POLES])


def contrasts(E, culture):
    sims = E @ CM.T
    ips = sims - sims.mean(axis=1, keepdims=True)
    out = {}
    for name, a, b in DIMS:
        s = ips[:, POLES.index(a)] - ips[:, POLES.index(b)]
        out[name] = (s[culture == "zh"].mean() - s[culture == "west"].mean()) * 1000
    return out


def chunk_mean_batched(texts, size):
    """所有文本的所有块 → 一次批量编码 → 按块长加权聚合成文本向量。"""
    pieces, owner, wlen = [], [], []
    for i, t in enumerate(texts):
        t = str(t)
        ps = [t[j:j + size] for j in range(0, max(len(t), 1), size)] or [t]
        for p in ps:
            pieces.append(p); owner.append(i); wlen.append(len(p))
    print(f"    分块：{len(texts)} 篇 -> {len(pieces)} 块（平均 {len(pieces)/len(texts):.2f} 块/篇）")
    E = emb(pieces, "passage: ")
    owner = np.array(owner)
    wlen = np.array(wlen, dtype=np.float32)
    out = np.zeros((len(texts), E.shape[1]), dtype=np.float32)
    np.add.at(out, owner, E * wlen[:, None])          # 块向量按块长加权累加
    wsum = np.zeros(len(texts), dtype=np.float32)
    np.add.at(wsum, owner, wlen)
    out = out / wsum[:, None]                          # 长度加权均值
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
    return out


rows = []
for label, s in [("媒体层·月份新闻", "news_aligned_month"),
                 ("大众层·同片影评", "reviews_samefilms_v3")]:
    df = pd.read_parquet(CORPUS / "analysis_sets" / f"{s}.parquet")
    n_each = args.per_set // 2
    sub = pd.concat([df[df.culture == "zh"].sample(n=n_each, random_state=7),
                     df[df.culture == "west"].sample(n=n_each, random_state=7)],
                    ignore_index=True)
    txt = sub["text"].astype(str).tolist()
    cult = sub["culture"].values
    print(f"\n{label}: 子样本 {len(sub)} 篇（两侧各 {n_each}）；"
          f"zh 字数中位 {int(sub[sub.culture=='zh']['text'].str.len().median())}，"
          f"west 字数中位 {int(sub[sub.culture=='west']['text'].str.len().median())}")
    a = contrasts(chunk_mean_batched(txt, args.chunk), cult)
    b = contrasts(emb([t[:args.chunk] for t in txt], "passage: "), cult)
    for name, _, _ in DIMS:
        rows.append({"层": label, "维度": name, "全文本(分块均值)": round(a[name], 2),
                     "截断(前2000字符)": round(b[name], 2),
                     "方向一致": "✓" if np.sign(a[name]) == np.sign(b[name]) else "✗"})

d = pd.DataFrame(rows)
print("\n" + d.to_string(index=False))
d.to_csv(CORPUS / "results" / "truncation_check.csv", index=False, encoding="utf-8-sig")
print("\n方向一致:", (d["方向一致"] == "✓").sum(), "/", len(d))
print("-> results/truncation_check.csv")
