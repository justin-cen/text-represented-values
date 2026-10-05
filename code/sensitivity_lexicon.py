# -*- coding: utf-8 -*-
"""词表敏感性分析：v1锚定 / v2锚定+扩充未筛 / v3冻结 三版词库对文本得分的影响。

对已有嵌入缓存的分析集，分别用三版词库质心计算 Hofstede 12 极相似度，
比较：(a) 每篇文本三版得分的相关；(b) 六维对比分 zh-west 方向的稳定性。

用法: python sensitivity_lexicon.py --set news_aligned_month
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
LEX = BASE / "work" / "lexicon"
CORPUS = BASE / "data_corpus"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PAIRS = {"IDV": ("IDV_col", "IDV_ind"), "PDI": ("PDI_high", "PDI_low"),
         "MAS": ("MAS_mas", "MAS_fem"), "UAI": ("UAI_high", "UAI_low"),
         "LTO": ("LTO_long", "LTO_short"), "IND": ("IND_res", "IND_ind")}


def load_versions():
    ha = json.load(open(LEX / "hofstede_anchor.json", encoding="utf-8"))
    ht = json.load(open(LEX / "hofstede_v2_translated_verified_e5.json", encoding="utf-8"))
    ex = json.load(open(BASE / "work" / "lexicon_expansion_hofstede.json", encoding="utf-8"))
    fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
    poles = [k for k in ha if k != "meta"]
    v1, v2, v3 = {}, {}, {}
    for p in poles:
        a_en, a_zh = ha[p]["anchor_en"], ht["zh"].get(p, [])
        v1[p] = a_en + a_zh
        v2[p] = v1[p] + ex.get(p, {}).get("en", []) + ex.get(p, {}).get("zh", [])
        fp = fr.get("hofstede", {}).get(p, {"en": a_en, "zh": a_zh})
        v3[p] = fp["en"] + fp["zh"]
    return {"v1_anchor": v1, "v2_anchor+exp": v2, "v3_frozen": v3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--device", default="cpu",
                    help="嵌入设备；GPU 散热异常期间一律用 cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU 线程上限")
    args = ap.parse_args()
    if args.device == "cpu":
        import torch
        torch.set_num_threads(args.threads)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=args.device)
    print(f"嵌入设备: {args.device}")

    emb = np.load(CORPUS / "emb_cache" / f"{args.set}_e5.npy").astype(np.float32)
    df = pd.read_parquet(CORPUS / "analysis_sets" / f"{args.set}.parquet")
    assert len(df) == emb.shape[0], f"缓存与语料行数不符 {emb.shape[0]} vs {len(df)}"

    versions = load_versions()
    all_sims = {}
    for vname, poles in versions.items():
        sims = {}
        for p, words in poles.items():
            words = [w for w in words if w and len(w) < 40]
            c = model.encode(["query: " + w for w in words], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=512, show_progress_bar=False)
            c = c.mean(axis=0)
            c /= np.linalg.norm(c)
            sims[p] = emb @ c
        all_sims[vname] = pd.DataFrame(sims)
        print(f"{vname}: 质心完成 {len(poles)} 极")

    # (a) 每极三版得分的文本级相关
    print("\n=== 文本级 Pearson 相关（v1~v3, v1~v2, v2~v3）===")
    poles = list(all_sims["v1_anchor"].columns)
    cor_tab = []
    for p in poles:
        a, b, c = (all_sims["v1_anchor"][p], all_sims["v2_anchor+exp"][p],
                   all_sims["v3_frozen"][p])
        cor_tab.append({"pole": p,
                        "r(v1,v3)": np.corrcoef(a, c)[0, 1],
                        "r(v1,v2)": np.corrcoef(a, b)[0, 1],
                        "r(v2,v3)": np.corrcoef(b, c)[0, 1]})
    cor = pd.DataFrame(cor_tab).round(3)
    print(cor.to_string(index=False))

    # (b) 六维对比分方向稳定性
    print("\n=== 六维对比分（zh-west, ×1000; 符号方向应跨版本一致）===")
    culture = df["culture"].values
    rows = []
    for dim, (x, y) in PAIRS.items():
        row = {"dim": dim}
        for vname in versions:
            d = all_sims[vname][x] - all_sims[vname][y]
            m = pd.Series(d).groupby(culture).mean()
            row[vname] = (m.get("zh", np.nan) - m.get("west", np.nan)) * 1000
        rows.append(row)
    stab = pd.DataFrame(rows).round(2)
    print(stab.to_string(index=False))
    out = CORPUS / "results" / f"{args.set}_lexicon_sensitivity.csv"
    cor.assign(note="文本级相关").to_csv(out, index=False, encoding="utf-8-sig")
    stab.to_csv(CORPUS / "results" / f"{args.set}_lexicon_sensitivity_contrasts.csv",
                index=False, encoding="utf-8-sig")
    print(f"-> {out} 及 _contrasts.csv")


if __name__ == "__main__":
    main()
