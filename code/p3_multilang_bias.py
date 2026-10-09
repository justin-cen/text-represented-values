# -*- coding: utf-8 -*-
"""P3 泛化检验：22 语种 Wikipedia 样本上的跨语言基线偏置。

检验：
(a) 语种层面是否存在系统性偏移（同一语种对所有刻面同向偏移）；
(b) 偏移幅度是否因语种而异 → 跨语言绝对相似度不可比；
(c) 文本内标准化后偏移是否被消除。
输出 results/p3_multilang_bias.csv + 控制台报告
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sims = pd.read_parquet(BASE / "data_corpus/results/wiki_multilang_facet_sims_v3.parquet")
print(f"载入 {len(sims):,} 行 | 语种 {sims.culture.nunique()}")

# 各语种 × 各刻面的平均绝对相似度
piv = sims.groupby(["culture", "facet"])["sim"].mean().unstack("facet")
print(f"矩阵: {piv.shape}")

# 语种偏移 = 该语种在所有刻面上的平均相似度 − 全局均值
grand = sims["sim"].mean()
off = (piv.mean(axis=1) - grand)
off = off.sort_values(ascending=False)
print(f"\n=== (a/b) 语种层面系统性偏移（所有刻面平均，绝对相似度）===")
print(f"全局均值: {grand:.4f} | 偏移极差: {off.max()-off.min():.4f}")
for lang, v in off.items():
    bar = "█" * int(abs(v) * 400)
    print(f"  {lang}  {v:+.4f}  {bar}")

# 语种内一致性：该语种各刻面偏移是否同向
print(f"\n=== 语种内偏移一致性（该语种所有刻面偏移的标准差 / 均值绝对值）===")
cons = []
for lang in piv.index:
    d = piv.loc[lang] - piv.mean(axis=0)      # 相对各刻面均值
    cons.append({"lang": lang, "mean_off": d.mean(), "sd_off": d.std(),
                 "同向比例": (np.sign(d) == np.sign(d.mean())).mean()})
C = pd.DataFrame(cons).sort_values("mean_off", ascending=False)
print(C.round(4).to_string(index=False))

# 文本内标准化后的偏移
sims["ips"] = sims["sim"] - sims.groupby("corpus_id")["sim"].transform("mean")
piv_ips = sims.groupby(["culture", "facet"])["ips"].mean().unstack("facet")
off_ips = (piv_ips.mean(axis=1) - sims["ips"].mean()).sort_values(ascending=False)
print(f"\n=== (c) 文本内标准化后语种偏移 ===")
print(f"偏移极差: {off_ips.max()-off_ips.min():.6f}（应≈0）")

out = pd.DataFrame({"lang": off.index, "绝对相似度偏移": off.values,
                    "标准化后偏移": off_ips.reindex(off.index).values}).merge(C, on="lang")
out.to_csv(BASE / "data_corpus/results/p3_multilang_bias.csv", index=False, encoding="utf-8-sig")
print("\n-> results/p3_multilang_bias.csv")
