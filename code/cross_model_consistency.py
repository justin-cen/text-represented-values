# -*- coding: utf-8 -*-
"""跨模型三角互证：e5 vs LaBSE 在同一语料上的一致性。

产出 results/cross_model.csv + 控制台摘要。
用法: python cross_model_consistency.py [--set reports_political]
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
DIMNAMES = [n for n, _, _ in DIMS]
POLES = [p for _, a, b in DIMS for p in (a, b)]


def contrast_per_text(sims):
    d = sims[sims.system == "hofstede"].copy()
    d["ips"] = d["sim"] - d.groupby("corpus_id")["sim"].transform("mean")
    w = d.groupby(["corpus_id", "culture", "facet"])["ips"].mean().unstack("facet")
    out = {}
    for name, a, b in DIMS:
        out[name] = w[a] - w[b]
    cc = pd.DataFrame(out).reset_index() * 1000 if False else pd.DataFrame(out).reset_index()
    cc[DIMNAMES] = cc[DIMNAMES] * 1000
    return cc


def group_contrast(cc):
    g = cc.groupby("culture")[DIMNAMES].mean()
    return (g.loc["zh"] - g.loc["west"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="reports_political")
    args = ap.parse_args()
    s = args.set

    e5 = pd.read_parquet(BASE / "results" / f"{s}_facet_sims_v3.parquet")
    lb = pd.read_parquet(BASE / "results" / f"{s}_facet_sims_v3_labse.parquet")
    ce, cl = contrast_per_text(e5), contrast_per_text(lb)
    common = ce["corpus_id"].isin(set(cl["corpus_id"]))
    ce = ce[common].reset_index(drop=True)
    cl = cl.set_index("corpus_id").loc[ce["corpus_id"]].reset_index()

    print(f"=== 跨模型一致性：{s}（e5 vs LaBSE, n={len(ce):,} 文本）===\n")

    # 1) 文本级相关（同一文本两模型得分的相关，按维度）
    print("【1】文本级对比分的跨模型相关（两模型是否给同一文本打同样的分）:")
    tl = {}
    for name in DIMNAMES:
        r = np.corrcoef(ce[name], cl[name])[0, 1]
        tl[name] = round(r, 3)
    print("   ", tl)

    # 2) 组间对比分（zh−west）两模型并列
    ge = group_contrast(ce)
    gl = group_contrast(cl)
    cmp = pd.DataFrame({"e5": ge.round(2), "LaBSE": gl.round(2)})
    cmp["方向一致"] = (np.sign(cmp["e5"]) == np.sign(cmp["LaBSE"])).map({True: "✓", False: "✗"})
    print("\n【2】组间对比分（zh−west, ×1000）两模型并列:")
    print(cmp.to_string())
    n_agree = (np.sign(cmp["e5"]) == np.sign(cmp["LaBSE"])).sum()
    print(f"\n方向一致维度数: {n_agree}/{len(DIMNAMES)}")

    # 3) 跨模型量级相关（两模型对“哪一维差异大”是否一致）
    rmag = np.corrcoef(cmp["e5"], cmp["LaBSE"])[0, 1]
    print(f"跨模型量级相关（对比分剖面）: r = {rmag:.3f}")

    cmp.insert(0, "set", s)
    cmp.insert(1, "dim", DIMNAMES)
    cmp.to_csv(BASE / "results" / f"cross_model_{s}.csv", encoding="utf-8-sig")
    print(f"\n-> results/cross_model_{s}.csv")
    print("\n判读：方向一致数越多越好；文本级相关 >0.5 表示两模型给同一文本打分趋势一致；"
          "量级相关 >0.8 表示两模型对差异结构认知一致。")


if __name__ == "__main__":
    main()
