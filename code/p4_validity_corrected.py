# -*- coding: utf-8 -*-
"""P4 收敛效度（含衰减校正）：WVS 省级均值受小样本噪声衰减，校正后估真实相关。

衰减校正：r_true = r_obs / sqrt(rel)，其中 rel = 省间真方差 /（省间真方差 + 省内抽样方差）。
输出 results/p4_validity_corrected.csv + 控制台报告。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

tp = pd.read_csv(BASE / "work/results/analysis/province_profiles.csv")
wvs = pd.read_csv(BASE / "work/data/ground_truth_wvs_china_provinces.csv")
wvs["province_cn"] = wvs["province"].str.split().str[-1]
wvs_de = wvs.drop(columns=["province"])
m = tp.merge(wvs_de, left_on="province", right_on="province_cn", how="inner")

print(f"合并省份 n={len(m)}；WVS 省级样本量 n: 中位={int(m.n.median())} 最小={int(m.n.min())} 最大={int(m.n.max())}")
print()

txt_cols = [c for c in tp.columns if c.endswith("_s") or c.endswith("_o")]
wvs_cols = [c for c in wvs_de.columns if c.endswith("_wvs")]

# 各 WVS 刻面的省级均值信度（基于省内抽样方差）
rel = {}
for c in wvs_cols:
    obs_var = m[c].var(ddof=1)                       # 省间观测方差
    # 抽样方差近似：用刻度范围估（WVS 多为 1-10 或 1-4 量表，取观测方差的 30% 作保守估计）
    rel[c] = 0.70                                     # 保守先验信度
# 更严谨：用 WVS 官方标准误不可得，改用"分半信度上界"法——以观测方差解释率估计
# 简化并保守：设省级均值信度 0.7（小样本省份 n≈35 时更低），给出校正区间
RELS = [0.5, 0.7, 0.9]

rows = []
for tc in txt_cols:
    for wc in wvs_cols:
        sub = m[[tc, wc]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub[tc], sub[wc])
        row = {"文本构念": tc, "WVS刻面": wc, "n": len(sub), "r": round(r, 3), "p": round(p, 4)}
        for rl in RELS:
            row[f"r校正(rel={rl})"] = round(min(r / np.sqrt(rl), 0.999), 3)
        rows.append(row)

R = pd.DataFrame(rows)
R.to_csv(BASE / "data_corpus/results/p4_validity_corrected.csv", index=False, encoding="utf-8-sig")

print("=== 衰减校正后最强收敛（按 r 校正 rel=0.7 排序，前 20）===")
top = R.sort_values("r校正(rel=0.7)", ascending=False).head(20)
for _, r in top.iterrows():
    star = "★" if r["p"] < 0.05 else ("†" if r["p"] < 0.10 else "")
    print(f"  {r['文本构念']:<24} ↔ {r['WVS刻面']:<22} r={r['r']:+.3f} → 校正 r={r['r校正(rel=0.7)']:+.3f} (p={r['p']:.3f}) {star}")

print()
sig = R[R["p"] < 0.05]
print(f"显著对（p<0.05）: {len(sig)}/{len(R)} = {len(sig)/len(R)*100:.1f}%（随机期望 5%）")
print(f"其中方向为正: {(sig['r']>0).sum()} / 为负: {(sig['r']<0).sum()}")
print()
print("=== 显著对对应的文本构念分布 ===")
print(sig.groupby("文本构念").size().sort_values(ascending=False).head(12).to_string())
print()
print("=== 显著对对应的 WVS 刻面分布 ===")
print(sig.groupby("WVS刻面").size().sort_values(ascending=False).to_string())
print("\n-> data_corpus/results/p4_validity_corrected.csv")
