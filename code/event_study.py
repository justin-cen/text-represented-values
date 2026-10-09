# -*- coding: utf-8 -*-
"""事件研究：最高政治话语层价值观语域对重大历史事件的结构性响应（1990—2026）。

方法：
1) 事件点标记 + 均值检验（事件前 vs 事件后窗口）；
2) 断点回归（year + post_event + trend）检验显著性。

事件：WTO 入世(2001)、北京奥运/金融危机(2008)、SCV 正式提出(2013)、新冠疫情(2020)。
输出 results/event_study.csv + 控制台报告。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = pd.read_csv(BASE / "results" / "political_yearpair.csv")
d = d[(d.year >= 1990) & (d.year <= 2026)].reset_index(drop=True)
print(f"年份范围: {d.year.min()}—{d.year.max()} (n={len(d)})")

EVENTS = [("WTO 入世", 2001), ("北京奥运/GFC", 2008),
          ("SCV 正式提出", 2013), ("新冠疫情", 2020)]
DIMS = list("PDI IDV MAS UAI LTO IND".split())
CN = {"PDI": "权力距离", "IDV": "个体主义", "MAS": "男性气质",
      "UAI": "不确定性规避", "LTO": "长/短期导向", "IND": "放纵/约束"}

rows = []
for ename, ey in EVENTS:
    for dim in DIMS:
        y = d[dim].values
        # 事件前后窗口（3 年）
        pre = d[(d.year >= ey - 3) & (d.year < ey)][dim].values
        post = d[(d.year >= ey) & (d.year <= ey + 2)][dim].values
        if len(pre) < 2 or len(post) < 2:
            continue
        diff = post.mean() - pre.mean()
        t, p = stats.ttest_ind(post, pre, equal_var=False)
        # 断点回归：y = a + b*post + c*year
        x_post = (d.year >= ey).astype(int)
        x_year = (d.year - 1990).astype(float)
        A = np.column_stack([np.ones(len(d)), x_year, x_post])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        resid = y - pred
        se = np.sqrt((resid @ resid) / (len(y) - 3))
        cov = se**2 * np.linalg.inv(A.T @ A)
        se_post = np.sqrt(cov[2, 2])
        t_post = coef[2] / se_post
        p_post = 2 * (1 - stats.t.cdf(abs(t_post), len(y) - 3))
        rows.append({"事件": ename, "事件年": ey, "维度": dim, "维度中文": CN[dim],
                     "事件前均值": round(pre.mean(), 2), "事件后均值": round(post.mean(), 2),
                     "变化Δ": round(diff, 2), "前后t": round(t, 2), "前后p": round(p, 4),
                     "断点回归b(post)": round(coef[2], 2), "断点t": round(t_post, 2),
                     "断点p": round(p_post, 4)})

r = pd.DataFrame(rows)
r.to_csv(BASE / "results" / "event_study.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== 事件前后窗口（±3 年）均值变化与断点回归 ===")
print(r[["事件", "事件年", "维度中文", "事件前均值", "事件后均值", "变化Δ",
         "前后t", "前后p", "断点t", "断点p"]].to_string(index=False))
print("\n断点回归 |t|>2 的显著变化:")
sig = r[np.abs(r["断点t"]) > 2]
print(sig[["事件", "事件年", "维度中文", "变化Δ", "断点t", "断点p"]].to_string(index=False))
print("\n-> results/event_study.csv")
