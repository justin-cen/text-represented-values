# -*- coding: utf-8 -*-
"""豆瓣×IMDB 同片 ID 直连配对集（年代对齐 ≤2011，tt 编号精确连接）。

链路: 豆瓣评论(Movie_Name) → 维表片名+又名消歧 → movie_id → 维表 IMDb tt → IMDB 评论
年代对齐: 豆瓣评论限 date ≤ 2011-12-31（IMDB 50k 采集于 2011 年）

输出: analysis_sets/reviews_douban_imdb_matched.parquet + _films.csv
"""
import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CJK_PREFIX = re.compile(r"^([\u4e00-\u9fff：·、，。《》！？（）\s]+?)\s*[A-Za-z]")


def norm(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
    return re.sub(r"^the ", "", s)


def zh_part(name):
    m = CJK_PREFIX.match(str(name))
    return (m.group(1).strip() if m else str(name).strip())


def latin_part(name, zp):
    return str(name)[len(zp):].strip() if zp and str(name).startswith(zp) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-per-film", type=int, default=3)
    ap.add_argument("--max-per-film", type=int, default=30)
    ap.add_argument("--min-review-len", type=int, default=50)
    ap.add_argument("--douban-until", default="2011-12-31")
    ap.add_argument("--west-file", default="reviews_en_imdb_films.parquet",
                    help="reviews_en 下的西方语料文件名（需含 tt 列）")
    ap.add_argument("--name", default="reviews_douban_imdb_matched",
                    help="输出集名（analysis_sets 下，不含扩展名）")
    ap.add_argument("--source", default="imdb_acl", help="西方侧 source 标签")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 维表：movie_id ↔ tt ----------
    info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                       usecols=["movie_name", "movie_id", "又名", "上映日期", "IMDb"],
                       dtype={"movie_id": str}, low_memory=False)
    info["movie_id"] = info["movie_id"].str.split(".").str[0]
    info = info[info["IMDb"].notna() & info["IMDb"].str.startswith("tt")].drop_duplicates("movie_id")
    info["db_year"] = pd.to_numeric(
        info["上映日期"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    info["n_name"] = info["movie_name"].map(norm)
    info["n_aka"] = info["又名"].fillna("").map(norm)
    tt_by_mid = dict(zip(info["movie_id"], info["IMDb"]))
    by_name = {}
    for _, r in info.iterrows():
        by_name.setdefault(r["n_name"], []).append((r["movie_id"], r["db_year"], r["n_aka"]))
    print(f"维表带 tt 的影片 {len(info):,}；唯一片名 {len(by_name):,}")

    # ---------- 豆瓣侧：片名消歧 → tt ----------
    rz = pd.read_parquet(BASE / "reviews_zh" / "reviews_zh.parquet",
                         columns=["item", "text", "date", "rating", "meta"])
    rz = rz[rz["text"].str.len() >= args.min_review_len]
    rz = rz[rz["date"] <= args.douban_until]   # 年代对齐
    print(f"豆瓣 ≤{args.douban_until} 且 ≥{args.min_review_len}字: {len(rz):,} 条")

    name2tt = {}
    for name in rz["item"].unique():
        zp = zh_part(name)
        lat = norm(latin_part(name, zp))
        cands = by_name.get(norm(zp)) or by_name.get(norm(name)) or []
        if not cands:
            continue
        if len(cands) == 1:
            tt = tt_by_mid.get(cands[0][0])
            if tt:
                name2tt[name] = (cands[0][0], tt)
        else:
            hit = [c for c in cands if lat and lat in c[2]]
            if len(hit) >= 1:
                tt = tt_by_mid.get(hit[0][0])
                if tt:
                    name2tt[name] = (hit[0][0], tt)
    print(f"豆瓣片名→tt 解析成功 {len(name2tt):,} 个")

    # ---------- 西方侧 ----------
    im = pd.read_parquet(BASE / "reviews_en" / args.west_file)
    im = im[im["text"].str.len() >= args.min_review_len]
    tt_counts = im.groupby("tt").size()
    print(f"西方侧带 tt 评论 {len(im):,} 条，影片 {len(tt_counts):,} 部（{args.west_file}）")

    # 预分组索引（避免逐片线性扫描）
    rz = rz.reset_index(drop=True)
    im = im.reset_index(drop=True)
    z_by_item = rz.groupby("item").indices
    w_by_tt = im.groupby("tt").indices

    # ---------- 配对 ----------
    parts, films_rows = [], []
    for name, (mid, tt) in name2tt.items():
        if name not in z_by_item or tt not in w_by_tt:
            continue
        zs = rz.iloc[z_by_item[name]]
        ls = im.iloc[w_by_tt[tt]]
        if len(zs) < args.min_per_film or len(ls) < args.min_per_film:
            continue
        zs = zs.sample(min(args.max_per_film, len(zs)), random_state=args.seed)
        ls = ls.sample(min(args.max_per_film, len(ls)), random_state=args.seed)
        n = min(len(zs), len(ls))
        za, la = zs.head(n), ls.head(n)
        parts.append(pd.DataFrame({
            "corpus_id": ["dbi_" + str(mid) + "_" + str(i) for i in range(n)],
            "culture": "zh", "genre": "review", "source": "douban",
            "channel": name, "item": name, "date": za["date"].values,
            "year": za["date"].str[:4].values, "rating": za["rating"].astype(float).values,
            "sentiment": None, "title": None, "text": za["text"].values,
            "meta": za["meta"].values, "matched_film": tt}))
        parts.append(pd.DataFrame({
            "corpus_id": la["corpus_id"].values, "culture": "west", "genre": "review",
            "source": args.source, "channel": la["tt"].values, "item": la["tt"].values,
            "date": None, "year": None,
            "rating": la["rating"].astype(float).values if "rating" in la else None,
            "sentiment": la["sentiment"].values if "sentiment" in la else None,
            "title": None, "text": la["text"].values,
            "meta": None, "matched_film": tt}))
        films_rows.append({"matched_film": tt, "douban_name": name, "movie_id": mid,
                           "n_zh": n, "n_west": n})
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(BASE / "analysis_sets" / f"{args.name}.parquet", index=False)
    pd.DataFrame(films_rows).to_csv(BASE / "analysis_sets" / f"{args.name}_films.csv",
                                    index=False, encoding="utf-8-sig")
    print(f"-> {args.name}.parquet  rows={len(res):,}  "
          f"zh={sum(res.culture=='zh'):,} west={sum(res.culture=='west'):,} "
          f"影片={res['matched_film'].nunique()}")


if __name__ == "__main__":
    main()
