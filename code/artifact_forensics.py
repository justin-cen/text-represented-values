# -*- coding: utf-8 -*-
"""测量伪影诊断（artifact forensics）：未校正的跨语言基线偏置如何系统性地伪造文化差异。

方法：对同一批中西新闻文本，分别用 (a) 未校正的绝对相似度、(b) 文本内标准化后的相似度
计算中西对比分，比较两者的方向差异——证明未校正会把全部刻面伪造成"中文更高"。

输出 results/artifact_forensics.csv + 控制台报告
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 新闻集（足够大、跨语言）
sims = pd.read_parquet(BASE / "results" / "news_aligned_month_facet_sims_v3.parquet")
if "culture" not in sims.columns:
    meta = pd.read_parquet(BASE / "analysis_sets" / "news_aligned_month.parquet")[["corpus_id", "culture"]]
    sims = sims.merge(meta, on="corpus_id", how="left")

rows = []
for system in sims.system.unique():
    sub = sims[sims.system == system].copy()
    # 未校正：绝对相似度
    raw = sub.groupby(["facet", "culture"])["sim"].mean().unstack("culture")
    raw["diff_raw"] = (raw.get("zh", 0) - raw.get("west", 0)) * 1000
    # 校正：文本内标准化
    sub["ips"] = sub["sim"] - sub.groupby("corpus_id")["sim"].transform("mean")
    ips = sub.groupby(["facet", "culture"])["ips"].mean().unstack("culture")
    ips["diff_ips"] = (ips.get("zh", 0) - ips.get("west", 0)) * 1000
    for facet in raw.index:
        rows.append({
            "体系": system, "刻面": facet,
            "未校正对比分(zh−west)": round(raw.loc[facet, "diff_raw"], 2),
            "校正后对比分(zh−west)": round(ips.loc[facet, "diff_ips"], 2) if facet in ips.index else np.nan,
        })

d = pd.DataFrame(rows)
d.to_csv(BASE / "results" / "artifact_forensics.csv", index=False, encoding="utf-8-sig")

print("=== 测量伪影诊断：未校正 vs 文本内标准化 ===")
for system, g in d.groupby("体系"):
    n = len(g)
    raw_zh_higher = (g["未校正对比分(zh−west)"] > 0).sum()
    ips_zh_higher = (g["校正后对比分(zh−west)"] > 0).sum()
    raw_pos_rate = raw_zh_higher / n * 100
    print(f"\n{system}（{n} 刻面）:")
    print(f"  未校正: {raw_zh_higher}/{n}（{raw_pos_rate:.0f}%）刻面呈'中文更高' —— 基线伪影")
    print(f"  校正后: {ips_zh_higher}/{n}（{ips_zh_higher/n*100:.0f}%）刻面呈'中文更高' —— 真实信号")
    bias = g["未校正对比分(zh−west)"].mean()
    print(f"  未校正的系统偏置均值: {bias:+.2f}（全刻面向中文系统性抬高）")

# 翻转统计：校正后方向与未校正相反的刻面比例
d["raw_sign"] = np.sign(d["未校正对比分(zh−west)"])
d["ips_sign"] = np.sign(d["校正后对比分(zh−west)"])
flip = (d["raw_sign"] != d["ips_sign"]).sum()
print(f"\n=== 关键结论 ===")
print(f"全部 {len(d)} 个刻面中：")
print(f"  未校正时呈'中文更高'的比例: {(d['raw_sign']>0).mean()*100:.1f}%")
print(f"  校正后呈'中文更高'的比例: {(d['ips_sign']>0).mean()*100:.1f}%")
print(f"  方向被基线偏置翻转的刻面: {flip}（{flip/len(d)*100:.1f}%）")
print("\n-> results/artifact_forensics.csv")
