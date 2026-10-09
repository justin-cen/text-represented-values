# -*- coding: utf-8 -*-
"""P1 增强①：事件研究的安慰剂检验——真实事件 vs 随机安慰剂事件年。

对每个可能的年份做同样的断点回归，看真实事件（2008/2020）是否位于分布的极端尾部。
输出 results/event_placebo.csv + 控制台报告。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = pd.read_csv(BASE / "data_corpus/results/political_yearpair.csv")
d = d[(d.year >= 1990) & (d.year <= 2026)].reset_index(drop=True)
DIMS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]
CN = {"PDI": "权力距离", "IDV": "个体主义", "MAS": "男性气质",
      "UAI": "不确定性规避", "LTO": "长/短期导向", "IND": "放纵/约束"}
REAL = {"2008": "北京奥运/GFC", "2020": "新冠疫情"}


def breakpoint_t(y, ey):
    x_post = (d.year >= ey).astype(int).values
    x_year = (d.year - 1990).astype(float).values
    A = np.column_stack([np.ones(len(y)), x_year, x_post])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = np.sqrt((resid @ resid) / (len(y) - 3))
    cov = se**2 * np.linalg.inv(A.T @ A)
    return coef[2] / np.sqrt(cov[2, 2])


rows = []
for dim in DIMS:
    y = d[dim].values
    # 所有可能的事件年（留出前后各 3 年窗口）
    cands = [yr for yr in range(1993, 2024)]
    ts = {yr: breakpoint_t(y, yr) for yr in cands}
    for yr, t in ts.items():
        rows.append({"维度": dim, "维度中文": CN[dim], "事件年": yr,
                     "断点t": round(t, 3), "真实事件": REAL.get(str(yr), "")})

R = pd.DataFrame(rows)
R.to_csv(BASE / "data_corpus/results/event_placebo.csv", index=False, encoding="utf-8-sig")

print("=== 事件研究安慰剂检验 ===")
print("对 1993—2023 每个年份做同样断点回归，看真实事件是否在 |t| 分布的极端尾部\n")
for ev_year, ev_name in [(2008, "北京奥运/GFC"), (2020, "新冠疫情")]:
    print(f"【{ev_name}（{ev_year}）】")
    for dim in DIMS:
        sub = R[R.维度 == dim]
        t_real = sub[sub.事件年 == ev_year]["断点t"].iloc[0]
        pct = (sub["断点t"].abs() < abs(t_real)).mean() * 100
        rank = (sub["断点t"].abs() > abs(t_real)).sum() + 1
        flag = "★ 极端" if pct >= 90 else ("† 偏高" if pct >= 75 else "")
        print(f"  {CN[dim]:<12} 断点t={t_real:+6.2f} | 在 {len(sub)} 个安慰剂年份中排第 {rank} "
              f"（超过 {pct:.0f}% 的安慰剂）{flag}")
    print()
