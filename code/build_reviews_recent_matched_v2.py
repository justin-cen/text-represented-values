# -*- coding: utf-8 -*-
"""近期同片配对集 v2：ID 一致性过滤 + Letterboxd 年度消歧。

输入:
  raw/douban_recent/reviews_recent.jsonl  (现抓, film_id 为豆瓣 subject_id)
  raw/douban_recent/id_map.csv           (douban_name -> movie_id, how)
  raw/douban_info/douban_movie_info.csv  (movie_id -> 上映日期年)
  reviews_en/reviews_en_letterboxd.parquet
输出:
  analysis_sets/reviews_samefilms_recent_v2.parquet + _films.csv
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT_DIR = BASE / "raw" / "douban_recent"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return re.sub(r"^the ", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-film", type=int, default=3)
    ap.add_argument("--max-per-film", type=int, default=30)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input", default="reviews_recent_2020plus.jsonl",
                    help="raw/douban_recent 下的输入文件名")
    args = ap.parse_args()

    # 维表年份
    info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                       usecols=["movie_id", "上映日期"], dtype={"movie_id": str},
                       low_memory=False)
    info["movie_id"] = info["movie_id"].str.split(".").str[0]
    info = info.drop_duplicates("movie_id")
    info["db_year"] = pd.to_numeric(
        info["上映日期"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    y_by_id = dict(zip(info["movie_id"], info["db_year"]))

    idmap = pd.read_csv(OUT_DIR / "id_map.csv", dtype=str).fillna("")
    idmap["movie_id"] = idmap["movie_id"].str.split(".").str[0]
    idmap = idmap[idmap["movie_id"].str.len() > 0]
    expect_id = dict(zip(idmap["douban_name"], idmap["movie_id"]))
    expect_key = dict(zip(idmap["douban_name"], idmap["matched_film"]))

    rows = [json.loads(l) for l in open(OUT_DIR / args.input, encoding="utf-8")
            if l.strip()]
    z = pd.DataFrame(rows)
    n0 = len(z)
    # ID 一致性过滤（剔除错误解析抓来的"别的片"评论）
    z["expect_id"] = z["douban_name"].map(expect_id)
    z = z[z["film_id"].astype(str) == z["expect_id"]]
    n1 = len(z)
    z = z[z["text"].str.len() >= args.min_len].drop_duplicates(subset=["text"])
    z["db_year"] = z["film_id"].astype(str).map(y_by_id)
    z["matched_film"] = z["douban_name"].map(expect_key)
    print(f"豆瓣近期: {n0:,} -> ID一致过滤 {n1:,} -> 长度/去重后 {len(z):,} 条, "
          f"影片 {z['matched_film'].nunique()} 部")

    lb = pd.read_parquet(BASE / "reviews_en" / "reviews_en_letterboxd.parquet")
    lb["key"] = lb["channel"].map(norm)
    lb["lb_year"] = pd.to_numeric(
        lb["meta"].str.extract(r'"film_year":\s*"(\d{4})"')[0], errors="coerce")

    parts, films_rows, n_conflict = [], [], 0
    for (key, y), zg in z.groupby(["matched_film", "db_year"]):
        if pd.isna(y):
            continue
        lc = lb[(lb["key"] == key) & (lb["lb_year"].notna()) &
                ((lb["lb_year"] - int(y)).abs() <= 1)]
        if len(lc) < args.min_per_film or len(zg) < args.min_per_film:
            continue
        zs = zg.sample(min(args.max_per_film, len(zg)), random_state=args.seed)
        ls = lc.sample(min(args.max_per_film, len(lc)), random_state=args.seed)
        n = min(len(zs), len(ls))
        za, la = zs.head(n), ls.head(n)
        parts.append(pd.DataFrame({
            "corpus_id": ["db2_" + str(r["review_id"]) for _, r in za.iterrows()],
            "culture": "zh", "genre": "review", "source": "douban_recent",
            "channel": za["film_title"].values, "item": za["douban_name"].values,
            "date": za["date"].values, "year": za["date"].str[:4].values,
            "rating": za["rating"].astype(float).values, "sentiment": None,
            "title": None, "text": za["text"].values,
            "meta": [json.dumps({"vote_count": v, "ip_location": ip}, ensure_ascii=False)
                     for v, ip in zip(za["vote_count"], za["ip_location"])],
            "matched_film": key}))
        parts.append(pd.DataFrame({
            "corpus_id": la["corpus_id"].values, "culture": "west", "genre": "review",
            "source": "letterboxd", "channel": la["channel"].values,
            "item": la["item"].values, "date": None, "year": None, "rating": None,
            "sentiment": None, "title": None, "text": la["text"].values,
            "meta": la["meta"].values, "matched_film": key}))
        films_rows.append({"matched_film": key, "douban_name": za["douban_name"].iloc[0],
                           "movie_id": za["film_id"].iloc[0], "db_year": int(y),
                           "lb_year": int(lc["lb_year"].mode().iloc[0]), "n_zh": n, "n_west": n})
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(BASE / "analysis_sets" / "reviews_samefilms_recent_v3.parquet", index=False)
    pd.DataFrame(films_rows).to_csv(
        BASE / "analysis_sets" / "reviews_samefilms_recent_v3_films.csv",
        index=False, encoding="utf-8-sig")
    print(f"-> reviews_samefilms_recent_v3.parquet  rows={len(res):,}  "
          f"zh={sum(res.culture=='zh'):,} west={sum(res.culture=='west'):,} "
          f"影片={res['matched_film'].nunique()}")
    print("豆瓣侧年代分布:", res[res.culture == "zh"]["year"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
