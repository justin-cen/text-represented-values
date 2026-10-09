# -*- coding: utf-8 -*-
"""P4 省级双体裁测量与验证：
(1) 两体裁（政府报告 / 省级新闻）省级剖面；
(2) 两体裁间一致性（同一省份、不同体裁）；
(3) 各自与 WVS 省级均值的收敛效度（含衰减校正）；
(4) 方法比较：LLM 标注（既有 v3）vs 嵌入/词库（本文管线）。
输出 results/p4_dual_genre.csv, p4_dual_validity.csv
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_profile(setname):
    sims = pd.read_parquet(BASE / f"data_corpus/results/{setname}_facet_sims_v3.parquet")
    h = sims[sims.system == "hofstede"].copy()
    h["ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
    prof = h.groupby(["corpus_id", "facet"])["ips"].mean().unstack("facet")
    meta = pd.read_parquet(BASE / f"data_corpus/analysis_sets/{setname}.parquet")[["corpus_id", "channel"]]
    prof = prof.join(meta.set_index("corpus_id"))
    return prof.groupby("channel").mean()      # 省级均值


print("=== 载入省级剖面 ===")
news = load_profile("province_news")
gov = load_profile("gov_reports_trunc")
print(f"新闻剖面: {news.shape} | 政府报告剖面: {gov.shape}")

common = news.index.intersection(gov.index)
print(f"共有省份: {len(common)}")

# ---------- (2) 体裁间一致性 ----------
print("\n=== 体裁间一致性（同省 政府报告 vs 新闻）===")
rows = []
for f in news.columns:
    a, b = gov.loc[common, f], news.loc[common, f]
    r, p = stats.pearsonr(a, b)
    rows.append({"facet": f, "r_体裁间": round(r, 3), "p": round(p, 4)})
G = pd.DataFrame(rows).sort_values("r_体裁间", ascending=False)
G.to_csv(BASE / "data_corpus/results/p4_dual_genre.csv", index=False, encoding="utf-8-sig")
print(f"  12 极体裁间相关: 中位 {G['r_体裁间'].median():+.3f} | 最高 {G['r_体裁间'].max():+.3f} | 最低 {G['r_体裁间'].min():+.3f}")
print(f"  显著(p<0.05)对数: {(G['p']<0.05).sum()}/12")
print(G.head(6).to_string(index=False))

# ---------- (3) WVS 收敛效度（两体裁） ----------
wvs = pd.read_csv(BASE / "work/data/ground_truth_wvs_china_provinces.csv")
wvs["province_cn"] = wvs["province"].str.split().str[-1]
wvs = wvs.drop(columns=["province"]).set_index("province_cn")
wvs_cols = [c for c in wvs.columns if c.endswith("_wvs")]

out = []
for gname, prof in [("政府报告", gov), ("省级新闻", news)]:
    m = prof.join(wvs, how="inner")
    for f in prof.columns:
        for w in wvs_cols:
            sub = m[[f, w]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub[f], sub[w])
            out.append({"体裁": gname, "facet": f, "WVS刻面": w, "n": len(sub),
                        "r": round(r, 3), "p": round(p, 4),
                        "r校正0.7": round(min(r / np.sqrt(0.7), 0.999), 3)})
R = pd.DataFrame(out)
R.to_csv(BASE / "data_corpus/results/p4_dual_validity.csv", index=False, encoding="utf-8-sig")

print("\n=== WVS 收敛效度（两体裁对比）===")
for gname in ["政府报告", "省级新闻"]:
    sub = R[R.体裁 == gname]
    sig = sub[sub.p < 0.05]
    print(f"\n  【{gname}】共 {len(sub)} 对 | 显著 {len(sig)}（{len(sig)/len(sub)*100:.1f}%，随机 5%）")
    top = sub.sort_values("r校正0.7", ascending=False).head(5)
    for _, r in top.iterrows():
        star = "★" if r["p"] < 0.05 else ("†" if r["p"] < 0.10 else "")
        print(f"    {r['facet']:<16} ↔ {r['WVS刻面']:<22} r={r['r']:+.3f} → {r['r校正0.7']:+.3f} (p={r['p']:.3f}) {star}")
print("\n-> results/p4_dual_genre.csv, results/p4_dual_validity.csv")
