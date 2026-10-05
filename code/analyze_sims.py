# -*- coding: utf-8 -*-
"""对 facet_sims 结果做文化差异分析（ipsative + 维度对比分）。

用法: python analyze_sims.py --set reviews_samefilms [--system hofstede]
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CONTRASTS = {
    "hofstede": {"IDV": ("IDV_col", "IDV_ind", "集体主义←→个人主义"),
                 "PDI": ("PDI_high", "PDI_low", "高权力距离←→低权力距离"),
                 "MAS": ("MAS_mas", "MAS_fem", "男性气质←→女性气质"),
                 "UAI": ("UAI_high", "UAI_low", "高不确定性回避←→低"),
                 "LTO": ("LTO_long", "LTO_short", "长期导向←→短期导向"),
                 "IND": ("IND_res", "IND_ind", "约束←→放纵")},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--system", default="hofstede")
    ap.add_argument("--suffix", default="", help="词库版本后缀，如 _v3")
    args = ap.parse_args()

    sims = pd.read_parquet(BASE / "results" / f"{args.set}_facet_sims{args.suffix}.parquet")
    h = sims[sims.system == args.system].copy()

    # 1) 文本内相对凸显度（ipsative）
    h["sim_ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
    m = h.groupby(["facet", "culture"])["sim_ips"].mean().unstack()
    m["diff(zh-west)"] = m["zh"] - m["west"]
    print(f"=== {args.system} 文本内相对凸显度（ipsative, ×1000）===")
    print((m * 1000).round(2))

    # 2) 维度对比分
    if args.system in CONTRASTS:
        w = h.pivot_table(index=["corpus_id", "culture"], columns="facet", values="sim")
        print("\n=== 维度对比分（均值, ×1000）===")
        for dim, (a, b, label) in CONTRASTS[args.system].items():
            c = (w[a] - w[b]).groupby("culture").mean()
            za, zb = c.get("zh", float("nan")), c.get("west", float("nan"))
            # 配对影片上的差异显著性（若有 matched_film）
            print(f"{dim:4s} {label}: zh={za * 1000:+.2f}  west={zb * 1000:+.2f}  "
                  f"差={(za - zb) * 1000:+.2f}")

    # 3) 若有 matched_film：影片内文化差（控制影片内容）的配对检验
    if "matched_film" in sims.columns:
        from scipy import stats
        w2 = h.pivot_table(index=["matched_film", "culture", "corpus_id"],
                           columns="facet", values="sim").groupby(["matched_film", "culture"]).mean()
        print("\n=== 同片配对：影片级差异的配对 t 检验（zh-west, ×1000）===")
        for dim, (a, b, label) in CONTRASTS.get(args.system, {}).items():
            d = (w2[a] - w2[b]).unstack("culture")
            d = d.dropna()
            diff = d["zh"] - d["west"]
            t, p = stats.ttest_1samp(diff, 0)
            print(f"{dim:4s}: 平均影片内差={diff.mean() * 1000:+.2f}  "
                  f"t={t:+.2f}  p={p:.2e}  n片={len(d)}")


if __name__ == "__main__":
    main()
