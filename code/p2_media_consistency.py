# -*- coding: utf-8 -*-
"""P2 增强：媒体层的省域一致性——31 省省级媒体画像 vs 全国媒体（人民日报）画像。

若各省媒体与全国媒体高度一致，则"媒体层传递"结论具备省域普遍性。
输出 results/p2_media_province_consistency.csv + 控制台报告
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def scv_profile(setname, group_col, id_col="corpus_id"):
    sims = pd.read_parquet(BASE / f"data_corpus/results/{setname}_facet_sims_v3.parquet")
    sims = sims[sims.system == "scv"].copy()
    sims["ips"] = sims["sim"] - sims.groupby(id_col)["sim"].transform("mean")
    if group_col not in sims.columns:
        meta = pd.read_parquet(BASE / f"data_corpus/analysis_sets/{setname}.parquet")
        sims = sims.merge(meta[[id_col, group_col]], on=id_col, how="left")
    return sims.pivot_table(index=group_col, columns="facet", values="ips", aggfunc="mean")


# 全国媒体（人民日报 2025 严格同年集的中文侧）
nat = scv_profile("news_sameyear_2025", "culture")
nat_zh = nat.loc["zh"]
print("全国媒体（人民日报 2025）SCV 画像（前 5）:")
for k, v in nat_zh.sort_values(ascending=False).head(5).items():
    print(f"  {k:<20} {v*1000:+7.2f}")

# 省级媒体
prov = scv_profile("province_news", "channel")
print(f"\n省级媒体: {prov.shape[0]} 省")

# 各省与全国画像的相关
rows = []
for p in prov.index:
    r, pv = stats.pearsonr(prov.loc[p], nat_zh)
    rows.append({"省份": p, "与全国媒体相关r": round(r, 3), "p": round(pv, 4),
                 "n_articles": None})
R = pd.DataFrame(rows).sort_values("与全国媒体相关r", ascending=False)
R.to_csv(BASE / "data_corpus/results/p2_media_province_consistency.csv", index=False, encoding="utf-8-sig")

print("\n=== 各省媒体 SCV 画像与全国媒体画像的相关 ===")
print(f"  中位 r = {R['与全国媒体相关r'].median():.3f}")
print(f"  最低 r = {R['与全国媒体相关r'].min():.3f} | 最高 r = {R['与全国媒体相关r'].max():.3f}")
print(f"  r>0.5 的省份: {(R['与全国媒体相关r']>0.5).sum()}/{len(R)}")
print(f"  p<0.05 的省份: {(R['p']<0.05).sum()}/{len(R)}")
print("\n  前 6 省:")
print(R.head(6).to_string(index=False))
print("\n  后 6 省:")
print(R.tail(6).to_string(index=False))

# 省际两两相关（媒体层内部同质性）
M = prov.T.corr()
iu = np.triu_indices_from(M, k=1)
print(f"\n=== 省际两两相关（媒体层内部同质性）===")
print(f"  中位 r = {np.median(M.values[iu]):.3f} | 范围 {M.values[iu].min():.3f} — {M.values[iu].max():.3f}")
print("\n-> results/p2_media_province_consistency.csv")
