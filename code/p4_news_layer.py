# -*- coding: utf-8 -*-
"""P4 新闻层分析：31 省省级媒体（人民网省级频道 1,779 篇）的价值观画像与验证。

(a) 省级新闻画像（SCV 12 值 + Hofstede 12 极）；
(b) 省内一致性：各省内部离散度 vs 省间差异；
(c) 与 WVS 省级均值的收敛效度（含衰减校正）。
输出 results/p4_news_profiles.csv, p4_news_validity.csv
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sims = pd.read_parquet(BASE / "data_corpus/results/province_news_facet_sims_v3.parquet")
if "channel" not in sims.columns:
    meta = pd.read_parquet(BASE / "data_corpus/analysis_sets/province_news.parquet")[["corpus_id", "channel"]]
    sims = sims.merge(meta, on="corpus_id", how="left")
print(f"新闻语料: {sims.corpus_id.nunique()} 篇 × {sims.facet.nunique()} 刻面 | 省份 {sims.channel.nunique()}")

# ipsative（文本内标准化）
sims["ips"] = sims["sim"] - sims.groupby("corpus_id")["sim"].transform("mean")

# ---------- (a) 省级画像 ----------
scv = sims[sims.system == "scv"]
prof_scv = scv.pivot_table(index="channel", columns="facet", values="ips", aggfunc="mean")
hof = sims[sims.system == "hofstede"]
prof_hof = hof.pivot_table(index="channel", columns="facet", values="ips", aggfunc="mean")
prof = pd.concat([prof_scv, prof_hof], axis=1)
n_by = sims.groupby("channel")["corpus_id"].nunique()
prof["n_articles"] = n_by
prof.to_csv(BASE / "data_corpus/results/p4_news_profiles.csv", encoding="utf-8-sig")
print(f"\n省级画像: {prof.shape[0]} 省 × {prof.shape[1]-1} 刻面")

print("\n=== 全国均值最高的 SCV 值（省级媒体口径）===")
scv_mean = prof_scv.mean().sort_values(ascending=False)
for k, v in scv_mean.items():
    print(f"  {k:<20} {v*1000:+7.2f}")

# ---------- (b) 省内 vs 省间方差分解 ----------
print("\n=== (b) 方差分解：省间差异 / 省内差异 ===")
rows = []
for f in list(prof_scv.columns)[:12]:
    sub = sims[(sims.system == "scv") & (sims.facet == f)]
    grand = sub["ips"].mean()
    between = sub.groupby("channel")["ips"].mean().var(ddof=1)
    within = sub.groupby("channel")["ips"].var(ddof=1).mean()
    icc = between / (between + within) if (between + within) > 0 else np.nan
    rows.append({"facet": f, "省间方差": between, "省内方差": within, "ICC": icc})
V = pd.DataFrame(rows).sort_values("ICC", ascending=False)
print(V.round(4).to_string(index=False))
print(f"\n  ICC 中位: {V.ICC.median():.4f} | ICC>0.05 的比例: {(V.ICC>0.05).mean()*100:.0f}%")

# ---------- (c) WVS 收敛效度 ----------
wvs = pd.read_csv(BASE / "work/data/ground_truth_wvs_china_provinces.csv")
wvs["province_cn"] = wvs["province"].str.split().str[-1]
wvs = wvs.drop(columns=["province"]).set_index("province_cn")
wvs_cols = [c for c in wvs.columns if c.endswith("_wvs")]

out = []
allfacets = list(prof_scv.columns) + list(prof_hof.columns)
for f in allfacets:
    m = prof[[f]].join(wvs, how="inner")
    for w in wvs_cols:
        sub = m[[f, w]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.pearsonr(sub[f], sub[w])
        out.append({"facet": f, "WVS刻面": w, "n": len(sub), "r": round(r, 3), "p": round(p, 4),
                    "r校正0.7": round(min(r / np.sqrt(0.7), 0.999), 3)})
R = pd.DataFrame(out)
R.to_csv(BASE / "data_corpus/results/p4_news_validity.csv", index=False, encoding="utf-8-sig")
sig = R[R.p < 0.05]
print(f"\n=== (c) 省级新闻 × WVS 收敛效度 ===")
print(f"  共 {len(R)} 对 | 显著(p<0.05) {len(sig)}（{len(sig)/len(R)*100:.1f}%，随机期望 5%）")
print(f"  正 {(sig['r']>0).sum()} / 负 {(sig['r']<0).sum()}")
print("\n  最强收敛（衰减校正后前 10）:")
for _, r in R.sort_values("r校正0.7", ascending=False).head(10).iterrows():
    star = "★" if r["p"] < 0.05 else ("†" if r["p"] < 0.10 else "")
    print(f"    {r['facet']:<18} ↔ {r['WVS刻面']:<22} r={r['r']:+.3f} → {r['r校正0.7']:+.3f} (p={r['p']:.3f}) {star}")
print("\n-> results/p4_news_profiles.csv, results/p4_news_validity.csv")
