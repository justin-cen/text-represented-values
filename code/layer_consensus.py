# -*- coding: utf-8 -*-
"""三层共识量化（论文A 实证核心）：
量化"国家层（政府工作报告）—媒体层（官方媒体）—大众层（豆瓣影评）"三者在
社会主义核心价值观 12 值与 Hofstede 12 极上的画像一致性。

产出 results/layer_consensus.csv（层间相关矩阵）、results/layer_profiles.csv（三层画像明细）
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCV_ORDER = ["SCV_prosperity", "SCV_democracy", "SCV_civility", "SCV_harmony",
             "SCV_freedom", "SCV_equality", "SCV_justice", "SCV_ruleoflaw",
             "SCV_patriotism", "SCV_dedication", "SCV_integrity", "SCV_friendliness"]
SCV_CN = {"SCV_prosperity": "富强", "SCV_democracy": "民主", "SCV_civility": "文明",
          "SCV_harmony": "和谐", "SCV_freedom": "自由", "SCV_equality": "平等",
          "SCV_justice": "公正", "SCV_ruleoflaw": "法治", "SCV_patriotism": "爱国",
          "SCV_dedication": "敬业", "SCV_integrity": "诚信", "SCV_friendliness": "友善"}
POLES = ["PDI_high", "PDI_low", "IDV_ind", "IDV_col", "MAS_mas", "MAS_fem",
         "UAI_high", "UAI_low", "LTO_long", "LTO_short", "IND_ind", "IND_res"]
POLE_CN = dict(zip(POLES, ["高权力距离", "低权力距离", "个人主义", "集体主义", "男性气质", "女性气质",
                           "高不确定规避", "低不确定规避", "长期导向", "短期导向", "放纵", "约束"]))

LAYERS = [
    ("国家层·政府工作报告", "reports_political_facet_sims_v3.parquet", "reports_political.parquet", "gz"),
    ("媒体层·官方媒体", "news_aligned_month_facet_sims_v3.parquet", "news_aligned_month.parquet", "rmrb"),
    ("大众层·豆瓣影评", "reviews_samefilms_v3_facet_sims_v3.parquet", "reviews_samefilms_v3.parquet", "db"),
]


def ips_profile(facet_path, set_path, system, order):
    """某层中文侧在某体系上的文本内标准化画像（×1000）。"""
    d = pd.read_parquet(BASE / "results" / facet_path)
    d = d[d.system == system].copy()
    d["ips"] = d["sim"] - d.groupby("corpus_id")["sim"].transform("mean")
    if "culture" not in d.columns:                      # 缺 culture 时再从集里取
        meta = pd.read_parquet(BASE / "analysis_sets" / set_path)[["corpus_id", "culture"]]
        d = d.merge(meta, on="corpus_id", how="left")
    zh = d[d["culture"] == "zh"]
    return (zh.groupby("facet")["ips"].mean().reindex(order) * 1000), zh["corpus_id"].nunique()


scv_prof, hof_prof, rows = {}, {}, []
for label, fpath, spath, tag in LAYERS:
    s, n1 = ips_profile(fpath, spath, "scv", SCV_ORDER)
    h, _ = ips_profile(fpath, spath, "hofstede", POLES)
    scv_prof[label] = s
    hof_prof[label] = h
    rows.append({"层": label, "中文文本数": n1})
    print(f"{label}: n={n1}")

S = pd.DataFrame(scv_prof)      # 12 值 × 3 层
H = pd.DataFrame(hof_prof)      # 12 极 × 3 层

print("\n=== SCV 12 值三层画像（ipsative ×1000）===")
disp = S.copy()
disp.index = [SCV_CN[i] for i in disp.index]
print(disp.round(2).to_string())

print("\n=== 层间画像相关（SCV 12 值，Pearson r）===")
corr = S.corr(method="pearson")
print(corr.round(3).to_string())
tri = corr.values[np.triu_indices(3, 1)]
print(f"三层两两相关：{tri.round(3)} | 均值 r={tri.mean():.3f}")

print("\n=== 层间画像相关（Hofstede 12 极）===")
corr_h = H.corr()
print(corr_h.round(3).to_string())
tri_h = corr_h.values[np.triu_indices(3, 1)]
print(f"三层两两相关：{tri_h.round(3)} | 均值 r={tri_h.mean():.3f}")

# 层间 L1 距离（画像差异幅度，越小越一致）
print("\n=== 层间 L1 距离（SCV 画像，×1000）===")
for i in range(3):
    for j in range(i + 1, 3):
        a, b = S.columns[i], S.columns[j]
        print(f"  {a} vs {b}: L1={np.abs(S[a] - S[b]).sum():.1f}")

# 三层共同重心（各层 Top3）
print("\n=== 各层价值观重心 Top3 ===")
for c in S.columns:
    top = S[c].sort_values(ascending=False).head(3)
    print(f"  {c}: " + "、".join(f"{SCV_CN[i]}({v:+.1f})" for i, v in top.items()))

corr.to_csv(BASE / "results" / "layer_consensus.csv", encoding="utf-8-sig")
pd.concat([S.add_prefix("SCV_"), H.add_prefix("HOF_")], axis=1).to_csv(
    BASE / "results" / "layer_profiles.csv", encoding="utf-8-sig")
print("\n-> results/layer_consensus.csv, results/layer_profiles.csv")
