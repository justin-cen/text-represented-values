# -*- coding: utf-8 -*-
"""P4 核心验证：31 省文本画像 × WVS 省级金标准的收敛效度（H1）。

文本画像：显著性(_s)/方向(_o) 双评分，来自 368 份省级政府工作报告（31 省）。
金标准：WVS wave 7 中国省级均值（13 刻面）。
输出 results/p4_convergent_validity.csv + 控制台报告。
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
print(f"文本画像 {len(tp)} 省 × WVS {len(wvs)} 省 → 合并 {len(m)} 省")
print("合并省份:", " ".join(m["province"].tolist()))
print()

txt_s = [c for c in tp.columns if c.endswith("_s")]
txt_o = [c for c in tp.columns if c.endswith("_o")]
wvs_cols = [c for c in wvs_de.columns if c.endswith("_wvs")]
print(f"文本构念: 显著性 {len(txt_s)} 个 / 方向 {len(txt_o)} 个 | WVS 刻面 {len(wvs_cols)} 个")
print()

rows = []
for tc in txt_s + txt_o:
    for wc in wvs_cols:
        sub = m[[tc, wc]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub[tc], sub[wc])
        rho, prho = stats.spearmanr(sub[tc], sub[wc])
        rows.append({"文本构念": tc, "WVS刻面": wc, "n": len(sub),
                     "Pearson r": round(r, 3), "p": round(p, 4),
                     "Spearman ρ": round(rho, 3), "p_ρ": round(prho, 4)})

R = pd.DataFrame(rows)
R.to_csv(BASE / "data_corpus/results/p4_convergent_validity.csv", index=False, encoding="utf-8-sig")

# 每个文本构念的最强相关
print("=== 每个文本构念的最强 WVS 对应（|r| 最大）===")
best = R.loc[R.groupby("文本构念")["Pearson r"].apply(lambda s: s.abs().idxmax())]
best = best.sort_values("Pearson r", key=lambda s: s.abs(), ascending=False)
for _, r in best.iterrows():
    star = "★" if r["p"] < 0.05 else ("†" if r["p"] < 0.10 else "")
    print(f"  {r['文本构念']:<26} ↔ {r['WVS刻面']:<24} r={r['Pearson r']:+.3f} p={r['p']:.3f} {star}")

print()
n_sig = (R["p"] < 0.05).sum()
n_tot = len(R)
print(f"全部 {n_tot} 对相关中，p<0.05 的: {n_sig} 个（{n_sig/n_tot*100:.1f}%，随机期望 5%）")
print(f"|r|>0.4 且 p<0.05 的: {((R['Pearson r'].abs()>0.4) & (R['p']<0.05)).sum()} 个")
print("\n-> data_corpus/results/p4_convergent_validity.csv")
