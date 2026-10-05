# -*- coding: utf-8 -*-
"""最高政治话语层分析：分来源画像 + 按年配对 + 体裁比较 + 图8。

产出:
  results/political_analysis.csv      分来源×维度对比分
  results/political_yearpair.csv      GWR vs SOTU/KS 按年配对
  work/paper/figs/fig8_political.png  政治话语层画像对比图
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(r"E:\culture-difference\data_corpus")
FIGS = Path(r"E:\culture-difference\work\paper\figs")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

POLES = ["PDI_high", "PDI_low", "IDV_ind", "IDV_col", "MAS_mas", "MAS_fem",
         "UAI_high", "UAI_low", "LTO_long", "LTO_short", "IND_ind", "IND_res"]
DIMS = [("PDI", "PDI_high", "PDI_low"), ("IDV", "IDV_col", "IDV_ind"),
        ("MAS", "MAS_mas", "MAS_fem"), ("UAI", "UAI_high", "UAI_low"),
        ("LTO", "LTO_long", "LTO_short"), ("IND", "IND_res", "IND_ind")]
SCV_ORDER = ["SCV_prosperity", "SCV_democracy", "SCV_civility", "SCV_harmony",
             "SCV_freedom", "SCV_equality", "SCV_justice", "SCV_ruleoflaw",
             "SCV_patriotism", "SCV_dedication", "SCV_integrity", "SCV_friendliness"]
SCV_CN = {"SCV_prosperity": "富强", "SCV_democracy": "民主", "SCV_civility": "文明",
          "SCV_harmony": "和谐", "SCV_freedom": "自由", "SCV_equality": "平等",
          "SCV_justice": "公正", "SCV_ruleoflaw": "法治", "SCV_patriotism": "爱国",
          "SCV_dedication": "敬业", "SCV_integrity": "诚信", "SCV_friendliness": "友善"}


def load(setname):
    sims = pd.read_parquet(BASE / "results" / f"{setname}_facet_sims_v3.parquet")
    df = pd.read_parquet(BASE / "analysis_sets" / f"{setname}.parquet")
    return sims, df


def ipsative(sims):
    d = sims.copy()
    d["ips"] = d["sim"] - d.groupby("corpus_id")["sim"].transform("mean")
    return d


def contrast(d, dims=DIMS):
    """按 culture 算各维度对比分均值 (×1000)"""
    w = d[d.facet.isin([p for _, a, b in dims for p in (a, b)])]
    w = w.groupby(["culture", "facet"])["ips"].mean().unstack()
    out = {}
    for name, a, b in dims:
        out[name] = (w[a] - w[b]) * 1000
    return pd.DataFrame(out)


def main():
    sims, df = load("reports_political")
    sims["culture_src"] = sims["corpus_id"].map(
        df.set_index("corpus_id")["channel"])
    sims["year"] = sims["corpus_id"].map(df.set_index("corpus_id")["year"])
    d = ipsative(sims[sims.system == "hofstede"])

    # ---------- 1) 分来源画像（GWR vs SOTU vs KS vs SOTEU） ----------
    src_prof = d.groupby(["channel", "facet"])["ips"].mean().unstack().reindex(columns=POLES) * 1000
    print("=== 分来源 12 极画像（×1000）===")
    print(src_prof.round(1).to_string())

    # 分来源维度对比分
    def src_contrast(dd):
        w = dd.groupby(["channel", "facet"])["ips"].mean().unstack()
        o = {}
        for name, a, b in DIMS:
            o[name] = (w[a] - w[b]) * 1000
        return pd.DataFrame(o)
    sc = src_contrast(d)
    print("\n=== 分来源六维对比分（×1000）===")
    print(sc.round(2).to_string())
    sc.to_csv(BASE / "results" / "political_analysis.csv", encoding="utf-8-sig")

    # ---------- 2) 按年配对（GWR vs SOTU / GWR vs KS） ----------
    zh = d[d.culture == "zh"].copy()
    we = d[d.culture == "west"].copy()
    year_con = {}
    for name, a, b in DIMS:
        zh_y = zh[zh.facet.isin([a, b])].groupby(["year", "facet"])["ips"].mean().unstack()
        we_y = we[we.facet.isin([a, b])].groupby(["year", "facet"])["ips"].mean().unstack()
        common = zh_y.index.intersection(we_y.index)
        year_con[name] = ((zh_y.loc[common, a] - zh_y.loc[common, b]) -
                          (we_y.loc[common, a] - we_y.loc[common, b])) * 1000
    yc = pd.DataFrame(year_con)
    yc.to_csv(BASE / "results" / "political_yearpair.csv", encoding="utf-8-sig")
    print("\n=== 按年配对对比分（zh−west, ×1000，节选）===")
    print(yc.tail(12).round(2).to_string())
    print("\n按年配对均值:", yc.mean().round(2).to_dict())
    # 配对 t 检验（按年）
    from scipy import stats as sst
    print("\n按年配对 t 检验 (n=%d 年):" % len(yc.dropna()))
    for c in yc.columns:
        x = yc[c].dropna()
        t, p = sst.ttest_1samp(x, 0)
        print(f"  {c}: mean={x.mean():+.2f}  t={t:+.2f}  p={p:.2e}")

    # ---------- 3) SCV 分来源画像（功能同构检验） ----------
    dscv = ipsative(sims[sims.system == "scv"])
    scv_prof = dscv.groupby(["channel", "facet"])["ips"].mean().unstack().reindex(columns=SCV_ORDER) * 1000
    print("\n=== SCV 分来源画像（×1000）===")
    print(scv_prof.round(1).rename(columns=SCV_CN).to_string())

    # ---------- 4) 体裁比较（政治 vs 新闻 vs 影评） ----------
    genre_rows = {}
    for gname, gfile in [("政治话语", None), ("新闻", "news_aligned_month"), ("影评", "reviews_samefilms_v3")]:
        if gfile is None:
            dd = d
        else:
            gs, _ = load(gfile)
            dd = ipsative(gs[gs.system == "hofstede"])
        genre_rows[gname] = contrast(dd).loc["zh"] - contrast(dd).loc["west"] if "zh" in contrast(dd).index else None
    # 修正：contrast 已返回 zh/west 两列转置，直接算
    def genre_contrast(gfile):
        if gfile is None:
            dd = d
        else:
            gs, _ = load(gfile)
            dd = ipsative(gs[gs.system == "hofstede"])
        c = contrast(dd)
        return c.loc["zh"] - c.loc["west"]
    gcmp = pd.DataFrame({
        "政治话语": genre_contrast(None),
        "新闻": genre_contrast("news_aligned_month"),
        "影评": genre_contrast("reviews_samefilms_v3"),
    }).T[[x[0] for x in DIMS]]
    print("\n=== 三层体裁的六维对比分（zh−west, ×1000）===")
    print(gcmp.round(2).to_string())
    gcmp.to_csv(BASE / "results" / "political_genre_compare.csv", encoding="utf-8-sig")

    # ---------- 5) 图8 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
    # 左：分来源 SCV（GWR vs SOTU vs KS）
    ax = axes[0]
    x = np.arange(len(SCV_ORDER))
    pick = ["中国政府工作报告", "US State of the Union", "UK King's/Queen's Speech"]
    colors = ["#C0392B", "#2E5E8C", "#27AE60"]
    for i, (s, c) in enumerate(zip(pick, colors)):
        ax.bar(x + (i - 1) * 0.27, scv_prof.loc[s, SCV_ORDER], width=0.27, color=c,
               label={"中国政府工作报告": "中国 GWR", "US State of the Union": "美国 SOTU",
                      "UK King's/Queen's Speech": "英国国王演讲"}[s])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCV_CN[k] for k in SCV_ORDER], fontsize=9, rotation=45)
    ax.set_ylabel("ipsative 凸显度（×1000）")
    ax.set_title("图8a 社会主义核心价值观画像：中国 GWR vs 美国 SOTU vs 英国国王演讲", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    # 右：三层体裁的 IND 与 LTO
    ax = axes[1]
    for gname, c, mk in [("政治话语", "#C0392B", "o"), ("新闻", "#2E5E8C", "s"), ("影评", "#27AE60", "^")]:
        for dim in ["IND", "LTO", "UAI", "MAS"]:
            pass
        ax.scatter([gcmp.loc[gname, "LTO"]], [gcmp.loc[gname, "IND"]], color=c, marker=mk, s=90, label=gname)
    ax.axhline(0, color="#333", lw=0.6)
    ax.axvline(0, color="#333", lw=0.6)
    for gname, c in [("政治话语", "#C0392B"), ("新闻", "#2E5E8C"), ("影评", "#27AE60")]:
        ax.annotate(gname, (gcmp.loc[gname, "LTO"], gcmp.loc[gname, "IND"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("LTO 长期导向对比分（zh−west, ×1000）")
    ax.set_ylabel("IND 放纵/约束对比分（zh−west, ×1000）")
    ax.set_title("图8b 三层体裁的 IND × LTO 定位（均中文显著更低）", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_political.png", bbox_inches="tight")
    print("\nfig8 done ->", FIGS / "fig8_political.png")


if __name__ == "__main__":
    main()
