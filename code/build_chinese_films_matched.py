# -*- coding: utf-8 -*-
"""华语片同片配对集：豆瓣（中文观众）× Letterboxd（英文观众）评同一部中国片。

链: 豆瓣华语片（纯中文片名）→ 维表 又名取英文名 → Letterboxd 规范化片名×年份(±1) 匹配
输出: analysis_sets/reviews_chinesefilms_matched.parquet + _films.csv
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return re.sub(r"^the ", "", s)


def first_latin(aka):
    for seg in re.split(r"[/／]", str(aka)):
        seg = seg.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 '&:;,.!?()\-–]+", seg):
            return seg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-film", type=int, default=4)
    ap.add_argument("--max-per-film", type=int, default=30)
    ap.add_argument("--min-review-len", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 维表华语片（中国大陆/港台，又名含英文名）
    info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                       usecols=["movie_name", "movie_id", "又名", "上映日期", "制片国家/地区"],
                       dtype={"movie_id": str}, low_memory=False)
    info["movie_id"] = info["movie_id"].str.split(".").str[0]
    info["db_year"] = pd.to_numeric(
        info["上映日期"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    cn = info[info["制片国家/地区"].fillna("").str.contains("中国|大陆|香港|台湾")].copy()
    cn["en_title"] = cn["又名"].map(first_latin)
    cn = cn[cn["en_title"].notna()]
    cn["key"] = cn["en_title"].map(norm)
    cn = cn[cn["key"].str.len() >= 3].drop_duplicates("movie_name")
    print(f"维表华语片（含英文名）: {len(cn):,}")

    # 豆瓣侧
    rz = pd.read_parquet(BASE / "reviews_zh" / "reviews_zh.parquet",
                         columns=["item", "text", "date", "rating", "meta"])
    rz = rz[rz["text"].str.len() >= args.min_review_len]
    cn = cn[cn["movie_name"].isin(set(rz["item"]))]
    print(f"豆瓣有评论的华语片: {len(cn):,}")

    # Letterboxd 侧
    lb = pd.read_parquet(BASE / "reviews_en" / "reviews_en_letterboxd.parquet")
    lb["key"] = lb["channel"].map(norm)
    lb["lb_year"] = pd.to_numeric(
        lb["meta"].str.extract(r'"film_year":\s*"(\d{4})"')[0], errors="coerce")
    keys_hit = set(cn["key"]) & set(lb["key"])
    cn = cn[cn["key"].isin(keys_hit)]
    print(f"片名在 Letterboxd 出现的华语片: {len(cn):,}")

    rz = rz.reset_index(drop=True)
    lb = lb.reset_index(drop=True)
    z_by_item = rz.groupby("item").indices
    w_by_key = lb.groupby("key").indices

    parts, films_rows, n_year = [], [], 0
    for _, f in cn.iterrows():
        if f["movie_name"] not in z_by_item:
            continue
        zs = rz.iloc[z_by_item[f["movie_name"]]]
        ls = lb.iloc[w_by_key[f["key"]]]
        if pd.notna(f["db_year"]):
            ls2 = ls[(ls["lb_year"] - f["db_year"]).abs() <= 1]
            if len(ls2) == 0:
                n_year += 1
                continue
            ls = ls2
        if len(zs) < args.min_per_film or len(ls) < args.min_per_film:
            continue
        zs = zs.sample(min(args.max_per_film, len(zs)), random_state=args.seed)
        ls = ls.sample(min(args.max_per_film, len(ls)), random_state=args.seed)
        n = min(len(zs), len(ls))
        za, la = zs.head(n), ls.head(n)
        key = f["movie_name"]
        parts.append(pd.DataFrame({
            "corpus_id": ["cnf_" + str(f["movie_id"]) + "_" + str(i) for i in range(n)],
            "culture": "zh", "genre": "review", "source": "douban",
            "channel": f["movie_name"], "item": f["movie_name"],
            "date": za["date"].values, "year": za["date"].str[:4].values,
            "rating": za["rating"].astype(float).values, "sentiment": None,
            "title": None, "text": za["text"].values, "meta": za["meta"].values,
            "matched_film": key}))
        parts.append(pd.DataFrame({
            "corpus_id": la["corpus_id"].values, "culture": "west", "genre": "review",
            "source": "letterboxd", "channel": la["channel"].values,
            "item": la["item"].values, "date": None, "year": None, "rating": None,
            "sentiment": None, "title": None, "text": la["text"].values,
            "meta": la["meta"].values, "matched_film": key}))
        films_rows.append({"matched_film": key, "en_title": f["en_title"],
                           "movie_id": f["movie_id"], "db_year": int(f["db_year"]) if pd.notna(f["db_year"]) else None,
                           "n_zh": n, "n_west": n})
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(BASE / "analysis_sets" / "reviews_chinesefilms_matched.parquet", index=False)
    pd.DataFrame(films_rows).to_csv(BASE / "analysis_sets" / "reviews_chinesefilms_matched_films.csv",
                                    index=False, encoding="utf-8-sig")
    print(f"-> reviews_chinesefilms_matched.parquet  rows={len(res):,}  "
          f"zh={sum(res.culture=='zh'):,} west={sum(res.culture=='west'):,} "
          f"影片={res['matched_film'].nunique()}  年份冲突剔除={n_year}")
    print("示例:", [r["matched_film"] + "/" + r["en_title"] for r in films_rows[:8]])


if __name__ == "__main__":
    main()
