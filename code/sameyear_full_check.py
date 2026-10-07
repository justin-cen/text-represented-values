# -*- coding: utf-8 -*-
"""全量同年对齐（2025×2025, 23,688 篇）vs 相邻年份（2026×2025, 27,000 篇）对比。

输出 results/sameyear_full_check.csv + 控制台报告（含配对 t 检验）。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
DIM_CN = {"PDI": "权力距离", "IDV": "个体主义", "MAS": "男性气质",
          "UAI": "不确定性规避", "LTO": "长/短期导向", "IND": "放纵/约束"}


def load(setname, tag="_v3"):
    sims = pd.read_parquet(BASE / "results" / f"{setname}_facet_sims{tag}.parquet")
    h = sims[sims.system == "hofstede"].copy()
    h["ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
    prof = h.groupby(["corpus_id", "facet"])["ips"].mean().unstack("facet")
    meta = pd.read_parquet(BASE / "analysis_sets" / f"{setname}.parquet")[
        ["corpus_id", "culture", "match_month"]]
    prof = prof.join(meta.set_index("corpus_id"))
    for name, a, b in DIMS:
        prof[name] = prof[a] - prof[b]
    return prof


same = load("news_sameyear_2025")
adj = load("news_aligned_month")
print(f"同年集: {len(same):,} 篇（zh {sum(same.culture=='zh'):,} / west {sum(same.culture=='west'):,}）")
print(f"相邻集: {len(adj):,} 篇（zh {sum(adj.culture=='zh'):,} / west {sum(adj.culture=='west'):,}）")

rows = []
print(f"\n{'维度':<12}{'同年(2025×2025)':>17}{'t':>9}{'相邻(2026×2025)':>17}{'t':>9}{'一致':>6}")
for name, _, _ in DIMS:
    s = same[name] * 1000
    a = adj[name] * 1000
    sv = s[same.culture == "zh"].mean() - s[same.culture == "west"].mean()
    av = a[adj.culture == "zh"].mean() - a[adj.culture == "west"].mean()
    st, sp = stats.ttest_ind(s[same.culture == "zh"], s[same.culture == "west"], equal_var=False)
    at, ap = stats.ttest_ind(a[adj.culture == "zh"], a[adj.culture == "west"], equal_var=False)
    ok = "✓" if np.sign(sv) == np.sign(av) else "✗"
    rows.append({"维度": name, "维度中文": DIM_CN[name],
                 "同年对比分": round(sv, 2), "同年t": round(st, 1), "同年p": sp,
                 "相邻对比分": round(av, 2), "相邻t": round(at, 1), "相邻p": ap,
                 "方向一致": ok})
    print(f"{DIM_CN[name]:<12}{sv:>17.2f}{st:>9.1f}{av:>17.2f}{at:>9.1f}{ok:>6}")

d = pd.DataFrame(rows)
d.to_csv(BASE / "results" / "sameyear_full_check.csv", index=False, encoding="utf-8-sig")
print(f"\n方向一致: {(d['方向一致']=='✓').sum()}/6")
print("-> results/sameyear_full_check.csv")
