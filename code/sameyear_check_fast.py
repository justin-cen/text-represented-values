# -*- coding: utf-8 -*-
"""同年对齐 vs 相邻年份对齐（新闻层）方向对照。

子样本：月份分层各取 n/side，同年集（人民日报2025×FineNews2025）与相邻集（2026×2025）
使用同一批 FineNews 西方文本口径，比较六维对比分方向。
输出 results/sameyear_check.csv
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
ap.add_argument("--per-side", type=int, default=220)
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


def contrasts(sub):
    txt = [str(t)[:2000] for t in sub["text"]]
    E = emb(txt, "passage: ")
    sims = E @ CM.T
    ips = sims - sims.mean(axis=1, keepdims=True)
    cult = sub["culture"].values
    out = {}
    for name, a, b in DIMS:
        s = ips[:, POLES.index(a)] - ips[:, POLES.index(b)]
        out[name] = (s[cult == "zh"].mean() - s[cult == "west"].mean()) * 1000
    return out


def month_stratified(df, per_side, seed=11):
    parts = []
    for mm, g in df.groupby(df["match_month"].astype(str).str.zfill(2)):
        for c in ("zh", "west"):
            pool = g[g["culture"] == c]
            if len(pool):
                parts.append(pool.sample(n=min(per_side // 9 + 1, len(pool)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


print("=== 同年对齐 vs 相邻年份对齐（新闻层）===")
same_df = pd.read_parquet(CORPUS / "analysis_sets" / "news_sameyear_2025.parquet")
adj_df = pd.read_parquet(CORPUS / "analysis_sets" / "news_aligned_month.parquet")
s_sub = month_stratified(same_df, args.per_side)
a_sub = month_stratified(adj_df, args.per_side)
print(f"同年集子样本 {len(s_sub)}（zh {sum(s_sub.culture=='zh')} / west {sum(s_sub.culture=='west')}）")
print(f"相邻集子样本 {len(a_sub)}（zh {sum(a_sub.culture=='zh')} / west {sum(a_sub.culture=='west')}）")
same = contrasts(s_sub)
adj = contrasts(a_sub)
rows = []
for name, _, _ in DIMS:
    rows.append({"维度": name, "同年对齐(2025×2025)": round(same[name], 2),
                 "相邻年份(2026×2025)": round(adj[name], 2),
                 "方向一致": "✓" if np.sign(same[name]) == np.sign(adj[name]) else "✗"})
d = pd.DataFrame(rows)
print("\n" + d.to_string(index=False))
d.to_csv(CORPUS / "results" / "sameyear_check.csv", index=False, encoding="utf-8-sig")
print("\n方向一致:", (d["方向一致"] == "✓").sum(), "/", len(d))
print("-> results/sameyear_check.csv")
