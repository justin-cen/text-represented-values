# -*- coding: utf-8 -*-
"""题项锚定对照：用 VSM2013 问卷题项原文（+anchor_zh 中文锚词）作锚，
与冻结词表（双语词表）的 Hofstede 六维对比分做一致性检验。

产出 results/item_anchor_check.csv + 控制台摘要。
"""
import io
import json
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

DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
POLES = [p for _, a, b in DIMS for p in (a, b)]

anchor = json.load(open(LEX / "hofstede_anchor.json", encoding="utf-8"))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

def emb(texts, prefix="query: "):
    return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=128, show_progress_bar=False)

# ---- 题项锚定质心：英文题项原文 + 中文锚词 ----
item_cent = {}
for pole in POLES:
    v = anchor[pole]
    texts = [it["text"] for it in v.get("items", [])] + list(v.get("anchor_zh", []))
    E = emb(texts)
    c = E.mean(axis=0)
    item_cent[pole] = c / np.linalg.norm(c)
print(f"题项锚定质心: {len(item_cent)} 极（英文题项原文 + 中文锚词）")

# ---- 冻结词表质心（复用 frozen v3） ----
fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
frozen_cent = {}
words_all, wfac = [], []
for pole in POLES:
    ws = [x for x in (fr["hofstede"][pole]["en"] + fr["hofstede"][pole]["zh"]) if x and len(x) < 40]
    Ew = emb(ws)
    c = Ew.mean(axis=0)
    frozen_cent[pole] = c / np.linalg.norm(c)

# ---- 逐集计算 ----
SETS = [("大众层·同片影评", "reviews_samefilms_v3", "e5"),
        ("媒体层·月份新闻", "news_aligned_month", "e5"),
        ("国家层·政治施政", "reports_political", "cm_e5")]
rows = []
for label, setname, tag in SETS:
    emb_f = CORPUS / "emb_cache" / f"{setname}_{tag}.npy"
    ids_f = CORPUS / "emb_cache" / f"{setname}_{tag}_ids.txt"
    df = pd.read_parquet(CORPUS / "analysis_sets" / f"{setname}.parquet")
    E = np.load(emb_f).astype(np.float32)
    assert len(df) == E.shape[0], (setname, len(df), E.shape[0])
    culture = df["culture"].values
    # 两套锚的逐文本相似度 → ipsative → 对比分
    def pole_contrast(cents):
        sims = np.stack([E @ cents[p] for p in POLES], axis=1)  # (n, 12)
        ips = sims - sims.mean(axis=1, keepdims=True)
        out = {}
        for name, a, b in DIMS:
            ia, ib = POLES.index(a), POLES.index(b)
            out[name] = ips[:, ia] - ips[:, ib]
        return pd.DataFrame(out) * 1000
    ca = pole_contrast(item_cent)
    cf = pole_contrast(frozen_cent)
    for name, _, _ in DIMS:
        za = ca[name][culture == "zh"].mean() - ca[name][culture == "west"].mean()
        zf = cf[name][culture == "zh"].mean() - cf[name][culture == "west"].mean()
        r = np.corrcoef(ca[name], cf[name])[0, 1]
        agree = "✓" if np.sign(za) == np.sign(zf) else "✗"
        rows.append({"layer": label, "dim": name,
                     "item_anchor": round(za, 2), "frozen_lexicon": round(zf, 2),
                     "text_level_r": round(r, 3), "方向一致": agree})
    print(f"\n=== {label} ===")
    print(pd.DataFrame(rows)[pd.DataFrame(rows).layer == label].to_string(index=False))

res = pd.DataFrame(rows)
res.to_csv(CORPUS / "results" / "item_anchor_check.csv", index=False, encoding="utf-8-sig")
print("\n=== 汇总 ===")
agg = res.groupby("dim").agg(
    一致数=("方向一致", lambda s: (s == "✓").sum()),
    文本级相关均值=("text_level_r", "mean")).round(3)
print(agg.to_string())
print("\n-> results/item_anchor_check.csv")
print("判读：方向一致越多越好（题项锚与词表锚互相佐证）；文本级相关 >0.5 表明两锚给同一文本打分趋势一致。")
