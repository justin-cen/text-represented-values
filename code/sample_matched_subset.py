# -*- coding: utf-8 -*-
"""从大型配对集中抽取按影片分层的子样本（用于可控嵌入时长）。

用法: python sample_matched_subset.py --src reviews_douban_genome_matched --n-per-side 10000
"""
import argparse
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--n-per-side", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(BASE / "analysis_sets" / f"{args.src}.parquet")
    parts = []
    for cul in ("zh", "west"):
        sub = df[df.culture == cul]
        n_films = sub["matched_film"].nunique()
        alloc = max(2, int(round(args.n_per_side / n_films)))
        shuffled = sub.sample(frac=1, random_state=args.seed)
        per = shuffled.groupby("matched_film", group_keys=False).head(alloc)
        parts.append(per.sample(min(args.n_per_side, len(per)), random_state=args.seed))
    res = pd.concat(parts).sample(frac=1, random_state=args.seed)
    out = BASE / "analysis_sets" / f"{args.src}_sample.parquet"
    res.to_parquet(out, index=False)
    print(f"-> {out}  rows={len(res):,}  zh={sum(res.culture=='zh'):,} "
          f"west={sum(res.culture=='west'):,}  影片={res['matched_film'].nunique()}")


if __name__ == "__main__":
    main()
