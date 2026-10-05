# -*- coding: utf-8 -*-
"""构建"同一部电影 × 中西受众"严格配对影评集。

原理：豆瓣片名含原名（如 '阿丽塔：战斗天使 Alita: Battle Angel'），
Letterboxd 片名即英文原名；规范化后精确匹配片名，取两侧都有足量评论的影片，
每片每侧抽样等量评论，产出 analysis_sets/reviews_samefilms.parquet。

注意：豆瓣评论来自 2019-10 抓取，Letterboxd 来自 2025-07 抓取，
同一刺激、不同时代的受众话语——使用时应在论文中声明时代差异。

用法: python match_samefilms.py [--min-per-film 8] [--max-per-film 40] [--seed 42]
"""
import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "analysis_sets"
OUT.mkdir(exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'&:,;.!?()\-– ]{1,}")


def norm_title(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^the ", "", s)  # 'The Batman' 与 'Batman' 对齐
    return s


def douban_latin(movie_name: str) -> str | None:
    """取豆瓣片名中最长的拉丁字母串作为原名候选。"""
    cands = [m.group(0).strip() for m in LATIN_RUN.finditer(movie_name or "")]
    cands = [c for c in cands if len(c) >= 3]
    if not cands:
        return None
    return norm_title(max(cands, key=len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-film", type=int, default=8)
    ap.add_argument("--max-per-film", type=int, default=40)
    ap.add_argument("--min-review-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rz = pd.read_parquet(BASE / "reviews_zh" / "reviews_zh.parquet")
    rl = pd.read_parquet(BASE / "reviews_en" / "reviews_en_letterboxd.parquet")

    rz = rz[rz["text"].str.len() >= args.min_review_len].copy()
    rz["key"] = rz["item"].map(douban_latin)
    rz = rz[rz["key"].notna() & (rz["key"] != "")]
    print(f"豆瓣侧: {len(rz):,} 条长评论, 提取拉丁片名 {rz['key'].nunique():,} 个")

    rl["key"] = rl["channel"].map(norm_title)
    print(f"Letterboxd侧: {len(rl):,} 条, 影片 {rl['key'].nunique():,} 部")

    # 两侧都有 >=min_per_film 条评论的影片
    zc = rz["key"].value_counts()
    lc = rl["key"].value_counts()
    shared = sorted(set(zc[zc >= args.min_per_film].index) &
                    set(lc[lc >= args.min_per_film].index))
    print(f"配对成功影片: {len(shared):,} 部")

    parts = []
    for key in shared:
        z = rz[rz["key"] == key].sample(min(args.max_per_film, zc[key]),
                                        random_state=args.seed)
        l = rl[rl["key"] == key].sample(min(args.max_per_film, lc[key]),
                                        random_state=args.seed)
        n = min(len(z), len(l))
        parts.append(z.head(n).assign(matched_film=key))
        parts.append(l.head(n).assign(matched_film=key))
    res = pd.concat(parts, ignore_index=True)
    out = OUT / "reviews_samefilms.parquet"
    res.to_parquet(out, index=False)
    print(f"-> {out}  rows={len(res):,}  "
          f"zh={sum(res.culture=='zh'):,} west={sum(res.culture=='west'):,}")
    films = res.groupby("matched_film").size()
    print(f"影片数 {len(films):,}, 每片两侧合计评论 中位 {int(films.median())} "
          f"最多 {int(films.max())}")
    # 影片对照表
    zh_films = res[res.culture == "zh"].groupby("matched_film").agg(
        douban_name=("item", "first"), n_zh=("text", "size"))
    we_films = res[res.culture == "west"].groupby("matched_film").agg(
        letterboxd_item=("item", "first"), n_west=("text", "size"))
    ftab = zh_films.join(we_films).reset_index()
    ftab.to_csv(OUT / "reviews_samefilms_films.csv", index=False, encoding="utf-8-sig")
    print(f"影片对照表 -> {OUT / 'reviews_samefilms_films.csv'}")
    print("\n配对片名样例:")
    for k in shared[:20]:
        print("  ", k)


if __name__ == "__main__":
    main()
