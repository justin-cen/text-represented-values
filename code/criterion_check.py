# -*- coding: utf-8 -*-
"""效标关联验证（RQ3）：文本衍生维度得分 vs Hofstede 国别得分。

将 news/reviews 的刻面相似度结果按 outlet→国家 聚合为 CN/US/UK 三方，
计算六维对比分（如 IDV_col-IDV_ind），与 Hofstede 得分做方向一致性与秩相关检验。

用法: python criterion_check.py --set news_aligned_month [--system hofstede]
"""
import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CONTRASTS = {"IDV": ("IDV_col", "IDV_ind", "idv", True),   # Hofstede idv 高=个人主义, 对比分高=集体主义 → 方向应负相关
             "PDI": ("PDI_high", "PDI_low", "pdi", False),
             "MAS": ("MAS_mas", "MAS_fem", "mas", False),
             "UAI": ("UAI_high", "UAI_low", "uai", False),
             "LTO": ("LTO_long", "LTO_short", "ltowvs", False),
             "IND": ("IND_res", "IND_ind", "ivr", True)}    # ivr 高=放纵, 对比分高=约束 → 负相关


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--system", default="hofstede")
    ap.add_argument("--suffix", default="", help="词库版本后缀，如 _v3")
    args = ap.parse_args()

    va = json.load(open(BASE / "docs" / "validation_assets.json", encoding="utf-8"))
    oc, hs = va["outlet_country"], va["hofstede_scores"]
    sims = pd.read_parquet(BASE / "results" / f"{args.set}_facet_sims{args.suffix}.parquet")
    h = sims[sims.system == args.system].copy()
    h["country"] = h["channel"].map(oc).fillna(h["culture"].map({"zh": "CN", "west": "WEST"}))

    w = h.pivot_table(index=["country", "corpus_id"], columns="facet", values="sim")
    unmapped = h[h["country"].isin(["WEST"])]["channel"].value_counts()
    if len(unmapped):
        print(f"[提示] 未纳入国家映射的频道（归入WEST）: {unmapped.to_dict()}")
    print(f"=== {args.set}: 国家×维度 文本对比分（均值, ×1000）===")
    rows = {}
    for dim, (a, b, _, _) in CONTRASTS.items():
        s = (w[a] - w[b]).groupby("country").mean() * 1000
        rows[dim] = s
    T = pd.DataFrame(rows).round(2)
    print(T)

    print("\n=== 方向一致性检查（文本对比分排序 vs Hofstede 排序）===")
    for dim, (a, b, hcol, invert) in CONTRASTS.items():
        countries = [c for c in ["CN", "US", "UK"] if c in T.index]
        text_score = T.loc[countries, dim]
        hof = pd.Series({c: hs[c][hcol] for c in countries})
        text_aligned = -text_score if invert else text_score  # 统一到 Hofstede 方向
        rho = text_aligned.rank().corr(hof.rank())
        agree = "✓" if rho > 0.5 else ("~" if rho > 0 else "✗")
        print(f"{dim:4s} {agree} Spearman ρ={rho:+.2f}  "
              f"文本(对齐后)={dict(text_aligned.round(1))}  Hofstede={dict(hof)}")


if __name__ == "__main__":
    main()
