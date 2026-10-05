# -*- coding: utf-8 -*-
"""三模型一致性快速检验（子样本版）：e5 vs LaBSE vs BGE-M3。
各集取月份/影片分层子样本（默认 200 篇，两侧均衡），三模型批量嵌入后比较六维对比分方向。
输出 results/cross_model_3way.csv"""
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
MODELS = [("e5", "intfloat/multilingual-e5-large", "query: ", "passage: "),
          ("LaBSE", "sentence-transformers/LaBSE", "", ""),
          ("BGE-M3", "BAAI/bge-m3", "", "")]

ap = argparse.ArgumentParser()
ap.add_argument("--per-side", type=int, default=100)
args = ap.parse_args()

fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
from sentence_transformers import SentenceTransformer


def build_centroids(model, qp):
    cent = {}
    for p in POLES:
        ws = [x for x in (fr["hofstede"][p]["en"] + fr["hofstede"][p]["zh"]) if x and len(x) < 40]
        E = model.encode([qp + t for t in ws], normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=96, show_progress_bar=False)
        c = E.mean(axis=0)
        cent[p] = c / np.linalg.norm(c)
    return np.stack([cent[p] for p in POLES])


def contrasts(sub, model, CM, pp):
    E = model.encode([pp + str(t)[:2000] for t in sub["text"]], normalize_embeddings=True,
                     convert_to_numpy=True, batch_size=96, show_progress_bar=False)
    sims = E @ CM.T
    ips = sims - sims.mean(axis=1, keepdims=True)
    cult = sub["culture"].values
    out = {}
    for name, a, b in DIMS:
        s = ips[:, POLES.index(a)] - ips[:, POLES.index(b)]
        out[name] = (s[cult == "zh"].mean() - s[cult == "west"].mean()) * 1000
    return out


def subsample(df, per_side):
    parts = []
    for c in ("zh", "west"):
        pool = df[df["culture"] == c]
        parts.append(pool.sample(n=min(per_side, len(pool)), random_state=11))
    return pd.concat(parts, ignore_index=True)


SETS = [("同片影评", "reviews_samefilms_v3"), ("月份新闻", "news_aligned_month"),
        ("政治施政", "reports_political")]
results = {}
for mlabel, mname, qp, pp in MODELS:
    print(f"\n===== 模型 {mlabel} =====", flush=True)
    model = SentenceTransformer(mname, device="cpu")
    CM = build_centroids(model, qp)
    for slabel, s in SETS:
        df = pd.read_parquet(CORPUS / "analysis_sets" / f"{s}.parquet")
        sub = subsample(df, args.per_side)
        c = contrasts(sub, model, CM, pp)
        results[(slabel, mlabel)] = c
        print(f"  {slabel}（n={len(sub)}）: IND {c['IND']:+.2f}", flush=True)
    del model

print("\n=== 三模型方向一致性 ===")
rows = []
for name, _, _ in DIMS:
    row = {"维度": name}
    for slabel, _ in SETS:
        vals = [results[(slabel, m)][name] for m, _, _, _ in MODELS]
        row[slabel] = " / ".join(f"{v:+.2f}" for v in vals)
        signs = set(np.sign(vals))
        row[slabel + "·一致"] = "✓" if len(signs) == 1 else "✗"
    rows.append(row)
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(CORPUS / "results" / "cross_model_3way.csv", index=False, encoding="utf-8-sig")
print("\n每格为 e5 / LaBSE / BGE-M3；✓=三模型同号")
print("-> results/cross_model_3way.csv")
