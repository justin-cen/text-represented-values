# -*- coding: utf-8 -*-
"""验证政治层 IND 的 BGE-M3 翻转是否为截断伪影：BGE-M3 分块均值（全文本）重测。"""
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

fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cpu")

cent = {}
for p in POLES:
    ws = [x for x in (fr["hofstede"][p]["en"] + fr["hofstede"][p]["zh"]) if x and len(x) < 40]
    E = model.encode(ws, normalize_embeddings=True, convert_to_numpy=True, batch_size=96, show_progress_bar=False)
    c = E.mean(axis=0)
    cent[p] = c / np.linalg.norm(c)
CM = np.stack([cent[p] for p in POLES])

df = pd.read_parquet(CORPUS / "analysis_sets" / "reports_political.parquet")
texts = [str(t) for t in df["text"]]
print(f"政治集 n={len(texts)}，字数中位 {int(df['text'].str.len().median())}")

# 分块均值（2000 字符块，按块长加权）
pieces, owner, wlen = [], [], []
for i, t in enumerate(texts):
    ps = [t[j:j+2000] for j in range(0, max(len(t), 1), 2000)] or [t]
    for p in ps:
        pieces.append(p); owner.append(i); wlen.append(len(p))
print(f"分块：{len(texts)} 篇 -> {len(pieces)} 块")
E = model.encode(pieces, normalize_embeddings=True, convert_to_numpy=True, batch_size=96, show_progress_bar=False)
owner = np.array(owner); wlen = np.array(wlen, dtype=np.float32)
V = np.zeros((len(texts), E.shape[1]), dtype=np.float32)
np.add.at(V, owner, E * wlen[:, None])
ws = np.zeros(len(texts), dtype=np.float32); np.add.at(ws, owner, wlen)
V = V / ws[:, None]
V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

sims = V @ CM.T
ips = sims - sims.mean(axis=1, keepdims=True)
cult = df["culture"].values
print("\nBGE-M3 分块均值（全文本）政治层对比分：")
for name, a, b in DIMS:
    s = ips[:, POLES.index(a)] - ips[:, POLES.index(b)]
    v = (s[cult == "zh"].mean() - s[cult == "west"].mean()) * 1000
    print(f"  {name}: {v:+.2f}")
