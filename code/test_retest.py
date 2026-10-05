# -*- coding: utf-8 -*-
"""C2 文内分半一致性（test-retest）：政治长文档前后两半的价值观画像相关性。

把每份政治施政文本切成前后两半，分别用 e5 嵌入，计算同一文档两半画像的
Pearson 相关（信度下限的直观证据）。产出 results/test_retest.csv。
"""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE = Path(r"E:\culture-difference")
CORPUS = BASE / "data_corpus"
LEX = BASE / "work" / "lexicon"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
frozen = {s: {f: {"en": list(v["en"]), "zh": list(v["zh"])}
              for f, v in fr[s].items()} for s in fr if s != "meta"}

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

def emb(texts, prefix="query: "):
    return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=128, show_progress_bar=False)

# 刻面质心（Hofstede 12 极）
POLES = ["PDI_high","PDI_low","IDV_ind","IDV_col","MAS_mas","MAS_fem",
         "UAI_high","UAI_low","LTO_long","LTO_short","IND_ind","IND_res"]
words, wfac = [], []
for f in POLES:
    for x in frozen["hofstede"][f]["en"] + frozen["hofstede"][f]["zh"]:
        if x and len(x) < 40:
            words.append(x); wfac.append(f)
Ew = emb(words)
cent = {}
for f in POLES:
    idx = [i for i, ff in enumerate(wfac) if ff == f]
    v = Ew[idx].mean(axis=0)
    cent[f] = v / np.linalg.norm(v)

df = pd.read_parquet(CORPUS / "analysis_sets" / "reports_political.parquet")
rows = []
for _, r in df.iterrows():
    t = " ".join(str(r["text"]).split())
    half = len(t) // 2
    h1, h2 = t[:half], t[half:]
    e1 = emb(["passage: " + h1[:8000]])[0]
    e2 = emb(["passage: " + h2[:8000]])[0]
    # ipsative 画像（12 极）
    def profile(e):
        sims = {f: float(e @ cent[f]) for f in POLES}
        m = np.mean(list(sims.values()))
        return {f: sims[f] - m for f in POLES}
    p1, p2 = profile(e1), profile(e2)
    v1 = np.array([p1[f] for f in POLES])
    v2 = np.array([p2[f] for f in POLES])
    corr = np.corrcoef(v1, v2)[0, 1] if v1.std() > 0 and v2.std() > 0 else np.nan
    rows.append({"corpus_id": r["corpus_id"], "channel": r["channel"],
                 "culture": r["culture"], "year": r["year"], "split_half_r": corr})

res = pd.DataFrame(rows)
res.to_csv(CORPUS / "results" / "test_retest.csv", index=False, encoding="utf-8-sig")
print("=== 文内分半一致性（test-retest, 政治长文档 12 极画像前后两半相关）===")
print(res.groupby("channel")["split_half_r"].agg(["count", "mean", "min"]).round(3).to_string())
print("\n整体均值:", round(res["split_half_r"].mean(), 3),
      "（>0.6 视为测量可靠；越接近 1 越稳定）")
