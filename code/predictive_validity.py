# -*- coding: utf-8 -*-
"""预测效度检验：影评价值观语域（文本内标准化对比分）能否在控制情感极性后仍预测评分。

检验"文本表征的价值观"是否具有超越情感极性的预测效度——这是把测量从描述推向验证的关键一步。

方法：影片级聚合（每片 zh 侧均值）→ 评分 ~ 六维语域 + 情感 + 片长 + 热度（对数化）。
稳健：标准化系数 + OLS 稳健标准误。
输出 results/predictive_validity.csv + 控制台报告。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]
CN = {"PDI": "权力距离", "IDV": "个体主义", "MAS": "男性气质",
      "UAI": "不确定性规避", "LTO": "长/短期导向", "IND": "放纵/约束"}

# 影片级聚合：中文侧六维语域
sims = pd.read_parquet(BASE / "results" / "reviews_samefilms_v3_facet_sims_v3.parquet")
h = sims[sims.system == "hofstede"].copy()
h["ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
prof = h.groupby(["corpus_id", "facet"])["ips"].mean().unstack("facet")
meta = pd.read_parquet(BASE / "analysis_sets" / "reviews_samefilms_v3.parquet")[
    ["corpus_id", "culture", "rating", "matched_film", "text"]]
prof = prof.join(meta.set_index("corpus_id"))

POLES = {"PDI": ("PDI_high", "PDI_low"), "IDV": ("IDV_col", "IDV_ind"),
         "MAS": ("MAS_mas", "MAS_fem"), "UAI": ("UAI_high", "UAI_low"),
         "LTO": ("LTO_long", "LTO_short"), "IND": ("IND_res", "IND_ind")}
for name, (a, b) in POLES.items():
    prof[name] = prof[a] - prof[b]

# 文本情感代理（独立于评分）：常见正负向词计数
POS = set("好 棒 赞 精彩 喜欢 感动 惊喜 完美 优秀 值得 经典 感人 震撼 good great excellent love amazing wonderful beautiful perfect best brilliant".split())
NEG = set("差 烂 糟 失望 难看 无聊 拖沓 尴尬 烂尾 失败 平庸 崩坏 bad terrible awful boring worst poor horrible ugly dull stupid disaster".split())

def senti(text):
    t = str(text)
    p = sum(t.count(w) for w in POS)
    n = sum(t.count(w) for w in NEG)
    return (p - n) / (p + n + 1.0)

zh = prof[prof.culture == "zh"].copy()
zh["sentiment"] = zh.text.map(senti)
zh["textlen"] = zh.text.astype(str).str.len()
film = zh.groupby("matched_film").agg(
    n_rev=("rating", "size"), rating=("rating", "mean"),
    sentiment=("sentiment", "mean"), textlen=("textlen", "mean"),
    **{d: (d, "mean") for d in DIMS}).reset_index()
film = film[film.n_rev >= 3].copy()
film["log_n"] = np.log(film.n_rev)
print(f"影片数（≥3 条中文影评）: {len(film)}")

# 标准化
Y = (film.rating - film.rating.mean()) / film.rating.std()
Xcols = DIMS + ["sentiment", "textlen", "log_n"]
X = film[Xcols].copy()
X = (X - X.mean()) / X.std()

import statsmodels.api as sm
A = sm.add_constant(X)
res = sm.OLS(Y, A).fit(cov_type="HC1")
print(f"\n=== 评分 ~ 语域 + 情感 + 控制（n={len(film)}, R²={res.rsquared:.4f}）===")
out = []
for c in A.columns:
    coef, se, t, p = res.params[c], res.bse[c], res.tvalues[c], res.pvalues[c]
    out.append({"变量": CN.get(c, c), "标准化系数β": round(coef, 4),
                "稳健SE": round(se, 4), "t": round(t, 2), "p": round(p, 4),
                "显著": "★" if p < 0.05 else ""})
    if c != "const":
        print(f"  {CN.get(c,c):<12} β={coef:+.4f}  t={t:+.2f}  p={p:.4f} {'★' if p<0.05 else ''}")
pd.DataFrame(out).to_csv(BASE / "results" / "predictive_validity.csv",
                         index=False, encoding="utf-8-sig")

# 与仅含情感的模型对比（增量 R²）
res0 = sm.OLS(Y, sm.add_constant(X[["sentiment", "textlen", "log_n"]])).fit(cov_type="HC1")
print(f"\n仅含情感与控制: R²={res0.rsquared:.4f}")
print(f"加入语域后增量 R²: {res.rsquared - res0.rsquared:.4f}")
print("\n-> results/predictive_validity.csv")
