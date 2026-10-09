# -*- coding: utf-8 -*-
"""P4 H4 检验：方法比较——LLM 标注（既有 v3，显著度 s）vs 嵌入/词库测量（本文管线）。

同 31 省、同 12 个社会主义核心价值观，比较两种方法所得省级画像的一致性。
并检验：两法与 WVS 的收敛效度谁更强（H4 预期 LLM ≥ 嵌入 ≥ 词典）。
输出 results/p4_method_compare.csv + 控制台报告
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 1) LLM 标注（既有）
llm = pd.read_csv(BASE / "work/results/analysis/province_profiles.csv", index_col=0)
llm_scv = llm[[c for c in llm.columns if c.startswith("SCV_") and c.endswith("_s")]]
llm_scv.columns = [c.replace("SCV_", "").replace("_s", "") for c in llm_scv.columns]
print(f"LLM 标注: {llm_scv.shape}")

# 2) 嵌入测量（新）
emb = pd.read_csv(BASE / "data_corpus/results/p4_news_profiles.csv", index_col=0)
emb_scv = emb[[c for c in emb.columns if c.startswith("SCV_")]]
emb_scv.columns = [c.replace("SCV_", "") for c in emb_scv.columns]
print(f"嵌入测量: {emb_scv.shape}")

common_prov = llm_scv.index.intersection(emb_scv.index)
common_val = [c for c in llm_scv.columns if c in emb_scv.columns]
print(f"共有省份 {len(common_prov)} | 共有价值观 {len(common_val)}")

# 3) 逐价值观比较（省级相关）
rows = []
for v in common_val:
    a = llm_scv.loc[common_prov, v]
    b = emb_scv.loc[common_prov, v]
    r, p = stats.pearsonr(a, b)
    rho, prho = stats.spearmanr(a, b)
    rows.append({"价值观": v, "LLM标注均值": round(a.mean(), 3), "嵌入均值(×1000)": round(b.mean() * 1000, 2),
                 "省际相关r": round(r, 3), "p": round(p, 4), "Spearman ρ": round(rho, 3)})
R = pd.DataFrame(rows).sort_values("省际相关r", ascending=False)
R.to_csv(BASE / "data_corpus/results/p4_method_compare.csv", index=False, encoding="utf-8-sig")

print("\n=== 方法比较：LLM 标注 vs 嵌入测量（31 省 × 12 价值观）===")
print(R.to_string(index=False))
print(f"\n  省际相关：中位 r = {R['省际相关r'].median():+.3f} | "
      f"范围 {R['省际相关r'].min():+.3f} ~ {R['省际相关r'].max():+.3f}")
print(f"  r>0.3 的价值观: {(R['省际相关r']>0.3).sum()}/12 | r<0 的: {(R['省际相关r']<0).sum()}/12")

# 4) 两法各自的 WVS 收敛效度对比
wvs = pd.read_csv(BASE / "work/data/ground_truth_wvs_china_provinces.csv")
wvs["province_cn"] = wvs["province"].str.split().str[-1]
wvs = wvs.drop(columns=["province"]).set_index("province_cn")
wvs_cols = [c for c in wvs.columns if c.endswith("_wvs")]

def validity(prof, cols, label):
    out = []
    m0 = prof[cols].join(wvs, how="inner")
    for f in cols:
        for w in wvs_cols:
            sub = m0[[f, w]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub[f], sub[w])
            out.append({"方法": label, "构念": f, "WVS刻面": w, "r": r, "p": p})
    return pd.DataFrame(out)

V = pd.concat([validity(llm_scv, common_val, "LLM标注"), validity(emb_scv, common_val, "嵌入测量")])
print("\n=== 两法各自的 WVS 收敛效度 ===")
for lab in ["LLM标注", "嵌入测量"]:
    s = V[V.方法 == lab]
    sig = s[s.p < 0.05]
    print(f"  {lab}: {len(s)} 对 | 显著 {len(sig)}（{len(sig)/len(s)*100:.1f}%）| "
          f"最强 |r| = {s.r.abs().max():.3f}")
V.to_csv(BASE / "data_corpus/results/p4_method_wvs.csv", index=False, encoding="utf-8-sig")
print("\n-> results/p4_method_compare.csv, results/p4_method_wvs.csv")
