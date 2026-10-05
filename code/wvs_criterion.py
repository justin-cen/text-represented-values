# -*- coding: utf-8 -*-
"""效标关联（真实 WVS 调查基准版）：文本 WVS 刻面得分 vs WVS-7 国别均值。

文本侧: news_aligned_month_facet_sims_v3 (system=wvs, ipsative by country)
调查侧: work/data/ground_truth_wvs.csv (WVS Wave 7, 加权国家均值, 13 刻面)
输出: results/wvs_criterion_v3.csv + 控制台表格
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference\data_corpus")
GT = Path(r"E:\culture-difference\work\data\ground_truth_wvs.csv")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

va = json.load(open(BASE / "docs" / "validation_assets.json", encoding="utf-8"))
oc = va["outlet_country"]

sims = pd.read_parquet(BASE / "results" / "news_aligned_month_facet_sims_v3.parquet")
w = sims[sims.system == "wvs"].copy()
w["country"] = w["channel"].map(oc)
w = w[w["country"].isin(["CN", "US", "UK"])]
# 文本内标准化（13 刻面 ipsative）
w["ips"] = w["sim"] - w.groupby("corpus_id")["sim"].transform("mean")
text = w.groupby(["facet", "country"])["ips"].mean().unstack() * 1000

gt = pd.read_csv(GT).set_index("iso").loc[["CHN", "USA", "GBR"]]
gt.index = ["CN", "US", "UK"]
gt.columns = [c.replace("_wvs", "") if c.endswith("_wvs") else c for c in gt.columns]

rows = []
for f in text.index:
    if f not in gt.columns:
        continue
    t = text.loc[f, ["CN", "US", "UK"]]
    s = gt.loc[["CN", "US", "UK"], f]
    rho = stats.spearmanr(t.values, s.values).statistic
    zw_text = t["CN"] - t[["US", "UK"]].mean()
    zw_survey = s["CN"] - s[["US", "UK"]].mean()
    rows.append({
        "facet": f,
        "text_CN": round(t["CN"], 2), "text_US": round(t["US"], 2), "text_UK": round(t["UK"], 2),
        "survey_CN": round(s["CN"], 3), "survey_US": round(s["US"], 3), "survey_UK": round(s["UK"], 3),
        "rho(n=3)": round(rho, 1),
        "text_zh-west": round(zw_text, 2),
        "survey_CN-均值(US,UK)": round(zw_survey, 3),
        "方向一致": "✓" if np.sign(zw_text) == np.sign(zw_survey) else "✗",
    })
out = pd.DataFrame(rows)
out.to_csv(BASE / "results" / "wvs_criterion_v3.csv", index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
n_ok = (out["方向一致"] == "✓").sum()
print(f"\n方向一致刻面: {n_ok}/{len(out)}")
print("rho 分布:", out["rho(n=3)"].value_counts().to_dict())
