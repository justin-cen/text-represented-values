# -*- coding: utf-8 -*-
"""从统一语料中抽取论文分析集（中西配对抽样）。

产出三个分析集（固定随机种子，可复现）：
1. reviews_general.parquet     中西影评各 N 条（分层抽样）
2. reviews_westernfilms.parquet 豆瓣"西片"评论 vs IMDB 评论（可比刺激设计）
3. news_general.parquet        中西新闻各 N 条（中文按频道分层，英文按关键词主题桶匹配）

用法: python make_analysis_sets.py [--n-review 10000] [--n-news 20000] [--seed 42]
"""
import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "analysis_sets"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# BBC 文档 -> 中文六类主题桶（弱匹配，仅供分层；正式分析可用分类器替代）
TOPIC_KEYWORDS = {
    "时政": r"\b(government|minister|parliament|election|president|brexit|policy|"
            r"senate|congress|diplomat|referendum|mp\b|prime minister)\b",
    "社会": r"\b(police|court|crime|school|hospital|nhs|housing|community|"
            r"charity|immigration|welfare|protest|strike)\b",
    "财经": r"\b(economy|bank|market|stock|shares|business|trade|inflation|"
            r"unemployment|growth|budget|tax|investment|pound|euro)\b",
    "科技": r"\b(technology|internet|digital|software|science|research|"
            r"computer|smartphone|data|ai\b|robot|space|nasa|app)\b",
    "教育": r"\b(university|student|school|education|teacher|exam|college|"
            r"pupil|academy|degree)\b",
    "娱乐": r"\b(film|movie|music|festival|actor|actress|album|show|television|"
            r"tv\b|award|singer|concert|star)\b",
}


def bucket_bbc(texts: pd.Series) -> pd.Series:
    pats = {k: re.compile(v, re.I) for k, v in TOPIC_KEYWORDS.items()}
    out = []
    for t in texts:
        hit = None
        for k, p in pats.items():
            if p.search(t):
                hit = k
                break
        out.append(hit if hit else "其他")
    return pd.Series(out, index=texts.index)


def stratified(df: pd.Series, n: int, seed: int) -> pd.Index:
    """按 df(类别序列) 的比例分层抽 n 个索引（每类至少1，超额类按比例）。"""
    vc = df.value_counts()
    alloc = (vc / vc.sum() * n).round().astype(int).clip(lower=1)
    diff = n - alloc.sum()
    if diff != 0:  # 微调使总数恰为 n
        top = alloc.idxmax()
        alloc[top] += diff
    idx = []
    rng = {}
    for cat, k in alloc.items():
        pool = df[df == cat].index
        k = min(k, len(pool))
        idx.append(df[pool].sample(k, random_state=seed).index)
    return pd.Index([i for s in idx for i in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-review", type=int, default=10000)
    ap.add_argument("--n-news", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-review-len", type=int, default=200,
                    help="长影评字符数下限（匹配论文'长影评'设定）")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rz = pd.read_parquet(BASE / "reviews_zh" / "reviews_zh.parquet")
    re_ = pd.read_parquet(BASE / "reviews_en" / "reviews_en.parquet")
    nz = pd.read_parquet(BASE / "news_zh" / "news_zh.parquet")
    ne = pd.read_parquet(BASE / "news_en" / "news_en.parquet")

    # ---------- 1. 影评总体集 ----------
    z = rz[rz["text"].str.len() >= args.long_review_len]
    z_key = z["year"].fillna(0).astype(int).astype(str) + "_" + z["rating"].astype(int).astype(str)
    zi = stratified(z_key, args.n_review, args.seed)
    w_key = re_["sentiment"].fillna("unl") + "_" + (re_["text"].str.len() // 500).clip(0, 8).astype(str)
    wi = stratified(w_key, args.n_review, args.seed)
    rev = pd.concat([z.loc[zi], re_.loc[wi]], ignore_index=True)
    rev.to_parquet(OUT / "reviews_general.parquet", index=False)
    print(f"reviews_general: zh={len(zi):,} west={len(wi):,}")

    # ---------- 2. 西片匹配集（可比刺激） ----------
    western_mask = rz["item"].str.contains(r"[A-Za-z]{3,}", regex=True, na=False)
    zw = rz[western_mask & (rz["text"].str.len() >= args.long_review_len)]
    n2 = min(args.n_review, len(zw))
    zi2 = zw.sample(n2, random_state=args.seed).index
    wi2 = re_.sample(n2, random_state=args.seed).index
    rev2 = pd.concat([rz.loc[zi2], re_.loc[wi2]], ignore_index=True)
    rev2.to_parquet(OUT / "reviews_westernfilms.parquet", index=False)
    n_films = rz.loc[zi2, "item"].nunique()
    print(f"reviews_westernfilms: zh={len(zi2):,} (覆盖西片 {n_films:,} 部) west={len(wi2):,}")

    # ---------- 3. 新闻总体集 ----------
    zi3 = stratified(nz["channel"], args.n_news, args.seed)
    ne = ne.copy()
    print("  正在为 BBC 新闻打主题桶（关键词弱匹配）…")
    ne["topic"] = bucket_bbc(ne["text"])
    zh_prop = nz.loc[zi3, "channel"].value_counts(normalize=True)
    parts = []
    for cat, prop in zh_prop.items():
        k = int(round(prop * args.n_news))
        pool = ne[ne["topic"] == cat]
        if len(pool) == 0:
            pool = ne
        parts.append(pool.sample(min(k, len(pool)), random_state=args.seed))
    ne_s = pd.concat(parts)
    news = pd.concat([nz.loc[zi3], ne_s.drop(columns=["topic"])], ignore_index=True)
    news.to_parquet(OUT / "news_general.parquet", index=False)
    print(f"news_general: zh={len(zi3):,} west={len(ne_s):,}")
    print("  中文频道分布:", nz.loc[zi3, 'channel'].value_counts().to_dict())
    print("  英文主题桶分布:", ne_s['topic'].value_counts().to_dict())


if __name__ == "__main__":
    main()
