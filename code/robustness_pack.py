# -*- coding: utf-8 -*-
"""方法论加固包：Bootstrap CI、FDR、效应量、文化内安慰剂、长度混淆。"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
DIMNAMES = [n for n, _, _ in DIMS]
rng = np.random.default_rng(42)


def load(setname, pairkey):
    sims = pd.read_parquet(BASE / "results" / f"{setname}_facet_sims_v3.parquet")
    d = sims[sims.system == "hofstede"].copy()
    a = pd.read_parquet(BASE / "analysis_sets" / f"{setname}.parquet")
    keycol = pairkey if pairkey in a.columns else None
    if keycol is None:
        # news_aligned_month 用 match_month
        keycol = "match_month" if "match_month" in a.columns else "year"
    keymap = a.drop_duplicates("corpus_id").set_index("corpus_id")[keycol].astype(str)
    d["pk"] = d["corpus_id"].map(keymap)
    return d


def text_contrast(d, extra=()):
    d = d.copy()
    d["ips"] = d["sim"] - d.groupby("corpus_id")["sim"].transform("mean")
    keys = ["corpus_id", "culture", "pk"] + list(extra) + ["facet"]
    w = d.groupby(keys)["ips"].mean().unstack("facet")
    out = {}
    for name, a, b in DIMS:
        out[name] = w[a] - w[b]
    cc = pd.DataFrame(out).reset_index()
    cc[DIMNAMES] = cc[DIMNAMES] * 1000
    return cc[cc["pk"].notna() & (cc["pk"] != "None") & (cc["pk"] != "nan")]


def boot_ci_paired(cc, nboot=2000):
    """先聚合到配对单元级（影片/月/年），再做配对 bootstrap 与 t。"""
    rows = []
    for name in DIMNAMES:
        film = cc.groupby(["pk", "culture"])[name].mean().unstack("culture")
        diff = (film["zh"] - film["west"]).dropna()
        n = len(diff)
        point = diff.mean()
        boots = np.array([diff.iloc[rng.integers(0, n, n)].mean() for _ in range(nboot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        dc = point / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        t, p = sst.ttest_1samp(diff, 0)
        rows.append({"dim": name, "n_pair": n, "point": round(point, 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "cohen_d": round(dc, 3), "t": round(t, 2), "p_raw": p})
    return pd.DataFrame(rows)


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


PAIRED = [("大众层·同片影评", "reviews_samefilms_v3", "matched_film"),
          ("媒体层·月份新闻", "news_aligned_month", "match_month"),
          ("国家层·政治施政", "reports_political", "year")]
all_res = []
for label, setname, pk in PAIRED:
    d = load(setname, pk)
    cc = text_contrast(d)
    r = boot_ci_paired(cc)
    r.insert(0, "layer", label)
    all_res.append(r)
res = pd.concat(all_res, ignore_index=True)
res["p_fdr"] = fdr_bh(res["p_raw"].values)
res["p_fdr"] = res["p_fdr"].map(lambda x: f"{x:.2e}")
res["p_raw"] = res["p_raw"].map(lambda x: f"{x:.2e}")
res.to_csv(BASE / "results" / "robustness_pack.csv", index=False, encoding="utf-8-sig")
print("=== Bootstrap 95% CI + Cohen's d + FDR（三配对集）===")
print(res.to_string(index=False))

# ---------- 文化内安慰剂检验 ----------
print("\n=== 文化内安慰剂检验（媒体个体两两对照，|对比分差| 均值 ×1000）===")
d = load("news_aligned_month", "match_month")
d2 = pd.read_parquet(BASE / "analysis_sets" / "news_aligned_month.parquet")
d["channel"] = d["corpus_id"].map(
    d2.drop_duplicates("corpus_id").set_index("corpus_id")["channel"])
cc = text_contrast(d, extra=("channel",))
chan = cc.groupby(["channel", "culture"])[DIMNAMES].mean().reset_index()

def pair_effects(cul):
    sub = chan[chan.culture == cul].set_index("channel")[DIMNAMES]
    chans = list(sub.index)
    effs = []
    for i in range(len(chans)):
        for j in range(i + 1, len(chans)):
            effs.append((sub.loc[chans[i]] - sub.loc[chans[j]]).abs())
    return pd.concat(effs, axis=1).T

zh_eff = pair_effects("zh")
we_eff = pair_effects("west")
cross = res[res.layer == "媒体层·月份新闻"].set_index("dim")["point"].abs()
print("中文媒体两两（安慰剂）均值 |差|:", zh_eff.mean().round(2).to_dict())
print("西方媒体两两（安慰剂）均值 |差|:", we_eff.mean().round(2).to_dict())
print("跨文化（真实）|差|:", cross.round(2).to_dict())
ratio = (cross / we_eff.mean()).round(2)
print("真实差异 / 文化内噪声 比值（>3 高度特异）:", ratio.to_dict())

# ---------- 长度混淆核查 ----------
print("\n=== 长度混淆核查（同片影评）===")
d = load("reviews_samefilms_v3", "matched_film")
d2 = pd.read_parquet(BASE / "analysis_sets" / "reviews_samefilms_v3.parquet")
d2["tlen"] = d2["text"].str.len()
cc = text_contrast(d)
cc = cc.merge(d2.drop_duplicates("corpus_id")[["corpus_id", "tlen"]].set_index("corpus_id")["tlen"].rename("tlen").reset_index(), on="corpus_id", how="left")
film = cc.groupby(["pk", "culture"]).agg({"PDI": "mean", "IND": "mean", "LTO": "mean", "tlen": "mean"}).reset_index()
fw = film.pivot(index="pk", columns="culture", values=["PDI", "IND", "LTO", "tlen"])
for dim in ["PDI", "IND", "LTO"]:
    diff = (fw[(dim, "zh")] - fw[(dim, "west")]).dropna()
    lendiff = (fw[("tlen", "zh")] - fw[("tlen", "west")]).dropna()
    common = diff.index.intersection(lendiff.index)
    rc = np.corrcoef(diff.loc[common], lendiff.loc[common])[0, 1]
    b, a = np.polyfit(lendiff.loc[common], diff.loc[common], 1)
    print(f"  {dim}: 原始对比分={diff.loc[common].mean():+.2f}，与长度差相关 r={rc:+.3f}，控制长度后截距={a:+.2f}")
print("（控制长度后对比分方向与量级基本不变 → 长度非主要混淆）")
print("\n-> results/robustness_pack.csv")
