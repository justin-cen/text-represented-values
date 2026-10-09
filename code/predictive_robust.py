# -*- coding: utf-8 -*-
"""P1 增强②：预测效度的分半交叉验证与安慰剂结果检验。

(a) 分半：随机把影片分两半，一半估系数、另一半预测，看样本外 R²；
(b) 安慰剂结果：用同一组语域预测"与价值观无关"的结果（影评数对数），应无显著增量。
输出 results/predictive_robust.csv + 控制台报告。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]
POS = set("好 棒 赞 精彩 喜欢 感动 惊喜 完美 优秀 值得 经典 感人 震撼 good great excellent love amazing wonderful beautiful perfect best brilliant".split())
NEG = set("差 烂 糟 失望 难看 无聊 拖沓 尴尬 烂尾 失败 平庸 崩坏 bad terrible awful boring worst poor horrible ugly dull stupid disaster".split())

sims = pd.read_parquet(BASE / "data_corpus/results/reviews_samefilms_v3_facet_sims_v3.parquet")
h = sims[sims.system == "hofstede"].copy()
h["ips"] = h["sim"] - h.groupby("corpus_id")["sim"].transform("mean")
prof = h.groupby(["corpus_id", "facet"])["ips"].mean().unstack("facet")
meta = pd.read_parquet(BASE / "data_corpus/analysis_sets/reviews_samefilms_v3.parquet")[
    ["corpus_id", "culture", "rating", "matched_film", "text"]]
prof = prof.join(meta.set_index("corpus_id"))
POLES = {"PDI": ("PDI_high", "PDI_low"), "IDV": ("IDV_col", "IDV_ind"), "MAS": ("MAS_mas", "MAS_fem"),
         "UAI": ("UAI_high", "UAI_low"), "LTO": ("LTO_long", "LTO_short"), "IND": ("IND_res", "IND_ind")}
for k, (a, b) in POLES.items():
    prof[k] = prof[a] - prof[b]

zh = prof[prof.culture == "zh"].copy()


def senti(t):
    t = str(t)
    p = sum(t.count(w) for w in POS)
    n = sum(t.count(w) for w in NEG)
    return (p - n) / (p + n + 1.0)


zh["sentiment"] = zh.text.map(senti)
zh["textlen"] = zh.text.astype(str).str.len()
film = zh.groupby("matched_film").agg(n_rev=("rating", "size"), rating=("rating", "mean"),
                                      sentiment=("sentiment", "mean"), textlen=("textlen", "mean"),
                                      **{d: (d, "mean") for d in DIMS}).reset_index()
film = film[film.n_rev >= 3].copy()
film["log_n"] = np.log(film.n_rev)
film = film.dropna(subset=DIMS + ["rating", "sentiment", "textlen", "log_n"]).reset_index(drop=True)
print(f"影片数: {len(film)}")


def fit_pred(train, test, ycol, xcols):
    mu, sd = train[xcols].mean(), train[xcols].std().replace(0, 1.0)
    Xtr = (train[xcols] - mu) / sd
    Xte = (test[xcols] - mu) / sd
    ytr = (train[ycol] - train[ycol].mean()) / train[ycol].std()
    yte = (test[ycol] - test[ycol].mean()) / test[ycol].std()
    ok = np.isfinite(Xtr.values).all(1) & np.isfinite(ytr.values)
    m = sm.OLS(ytr[ok], sm.add_constant(Xtr[ok])).fit()
    okte = np.isfinite(Xte.values).all(1) & np.isfinite(yte.values)
    pred = m.predict(sm.add_constant(Xte[okte], has_constant="add"))
    yv = yte[okte]
    ss_res = ((yv - pred) ** 2).sum()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


X_ALL = DIMS + ["sentiment", "textlen", "log_n"]
X_BASE = ["sentiment", "textlen", "log_n"]

print("\n=== (a) 分半交叉验证（10 次随机二分，样本外 R²）===")
rng = np.random.default_rng(42)
rows = []
for i in range(10):
    idx = rng.permutation(len(film))
    tr, te = film.iloc[idx[: len(film) // 2]], film.iloc[idx[len(film) // 2:]]
    r_full = fit_pred(tr, te, "rating", X_ALL)
    r_base = fit_pred(tr, te, "rating", X_BASE)
    rows.append({"折": i + 1, "样本外R²_全模型": r_full, "样本外R²_仅情感": r_base,
                 "增量": r_full - r_base})
C = pd.DataFrame(rows)
print(C.round(4).to_string(index=False))
print(f"\n  样本外 R²（全模型）均值: {C['样本外R²_全模型'].mean():.4f}")
print(f"  样本外 R²（仅情感）均值: {C['样本外R²_仅情感'].mean():.4f}")
print(f"  **语域增量（样本外）均值: {C['增量'].mean():+.4f}**")

print("\n=== (b) 安慰剂结果：同样语域预测『影评条数(log)』 ===")
film["log_n_out"] = np.log(film.n_rev)
r_placebo_all = fit_pred(film.iloc[: len(film) // 2], film.iloc[len(film) // 2:], "log_n_out", DIMS + ["sentiment", "textlen"])
print(f"  用六维语域预测影评条数的样本外 R²: {r_placebo_all:.4f}（接近 0 或为负=无语域效应）")

C.to_csv(BASE / "data_corpus/results/predictive_robust.csv", index=False, encoding="utf-8-sig")
print("\n-> results/predictive_robust.csv")
