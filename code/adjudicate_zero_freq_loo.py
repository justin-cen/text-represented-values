# -*- coding: utf-8 -*-
"""51 词机器裁定 · 二级：留一法（LOO）影响检验。

对 full_freq < 10 的锚定词，逐一从所属刻面质心剔除，
在 news_aligned_month 嵌入缓存上重算该刻面的 zh−west 对比分：
  - sign_flip（翻转方向）或 |Δcontrast| > 0.5 × |contrast_with| → drop（噪声词）
  - 否则 → keep（该词几乎不影响测量，保留构念覆盖）
输出: work/lexicon/frozen_v3/adjudication_51_final.csv + lexicon_v3_1_frozen.json
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
LEXDIR = BASE / "work" / "lexicon" / "frozen_v3"
CORPUS = BASE / "data_corpus"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

stage1 = pd.read_csv(LEXDIR / "adjudication_51_stage1.csv")
cand = stage1[stage1.full_freq < 10].copy()
print(f"二级检验对象（full_freq<10）: {len(cand)} 个（真零频 {sum(stage1.full_freq==0)}）")

# 冻结词库
fr = json.load(open(LEXDIR / "lexicon_v3_frozen.json", encoding="utf-8"))
frozen = {s: {f: {"en": list(v["en"]), "zh": list(v["zh"])}
              for f, v in fr[s].items() if f != "meta"} for s in fr if s != "meta"}

# 嵌入模型（只算质心词）
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

def emb(texts):
    return model.encode(["query: " + t for t in texts], normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=256, show_progress_bar=False)

# 文本嵌入缓存 + culture
emb_cache = np.load(CORPUS / "emb_cache" / "news_aligned_month_e5.npy").astype(np.float32)
df_news = pd.read_parquet(CORPUS / "analysis_sets" / "news_aligned_month.parquet")
assert len(df_news) == emb_cache.shape[0]
culture = df_news["culture"].values

# 逐刻面构建质心（含/不含候选词），逐候选词算影响
rows = []
for (sysname, facet), g in cand.groupby(["system", "facet"]):
    if sysname not in frozen or facet not in frozen[sysname]:
        continue
    w_en = frozen[sysname][facet]["en"]
    w_zh = frozen[sysname][facet]["zh"]
    words_full = [x for x in (w_en + w_zh) if x and len(x) < 40]
    Ef = emb(words_full)
    c_full = Ef.mean(axis=0)
    c_full /= np.linalg.norm(c_full)
    sim_full = emb_cache @ c_full
    con_full = sim_full[culture == "zh"].mean() - sim_full[culture == "west"].mean()
    for _, r in g.iterrows():
        word = r["word"]
        keep_mask = [w != word for w in words_full]
        if sum(keep_mask) == 0:
            continue
        c_loo = Ef[keep_mask].mean(axis=0)
        c_loo /= np.linalg.norm(c_loo)
        sim_loo = emb_cache @ c_loo
        con_loo = sim_loo[culture == "zh"].mean() - sim_loo[culture == "west"].mean()
        d_cos = 1 - float(np.dot(c_full, c_loo))
        d_con = (con_full - con_loo) * 1000
        sign_flip = (con_full * con_loo) < 0
        mag = abs(d_con) > 0.5 * abs(con_full * 1000) if abs(con_full) > 1e-9 else False
        verdict = "drop" if (sign_flip or mag) else "keep"
        reason = ("翻转对比分方向" if sign_flip else
                  ("拉动幅度过半" if mag else "影响可忽略"))
        rows.append({"system": sysname, "facet": facet, "lang": r["lang"],
                     "word": word, "full_freq": int(r["full_freq"]),
                     "delta_centroid_cos": round(d_cos, 5),
                     "contrast_with": round(con_full * 1000, 3),
                     "contrast_without": round(con_loo * 1000, 3),
                     "delta_contrast": round(d_con, 3),
                     "sign_flip": sign_flip, "verdict": verdict, "reason": reason})
    print(f"  [{sysname}/{facet}] 完成 {len(g)} 词", flush=True)

out = pd.DataFrame(rows)
out.to_csv(LEXDIR / "adjudication_51_final.csv", index=False, encoding="utf-8-sig")
print("\n判定分布:", out["verdict"].value_counts().to_dict())
print("\n=== 判定明细 ===")
print(out.to_string(index=False))

# 生成 v3.1 冻结词库（剔除 drop 词 + full_freq>=10 的一律保留）
drops = set(out.loc[out.verdict == "drop", "word"].tolist())
frozen31 = {}
for s, facets in frozen.items():
    frozen31[s] = {}
    for f, v in facets.items():
        frozen31[s][f] = {"en": [w for w in v["en"] if w not in drops],
                          "zh": [w for w in v["zh"] if w not in drops]}
meta = {"version": "v3.1-adjudicated", "built": "2026-10-03",
        "rule": "51个抽样零频锚定词的机器裁定：full_freq<10且LOO翻转/拉动过半→drop（" +
                str(len(drops)) + "词），其余保留；full_freq>=10 的 6 词恢复保留",
        "dropped": sorted(drops)}
frozen31 = {"meta": meta, **frozen31}
json.dump(frozen31, open(LEXDIR / "lexicon_v3_1_frozen.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n-> lexicon_v3_1_frozen.json（drop {len(drops)} 词: {sorted(drops)}）")
