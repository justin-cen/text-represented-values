# -*- coding: utf-8 -*-
"""版本消歧重配对：片名 + 上映年份（±1）双重约束的"同片"配对集。

问题：同名影片（重拍片/剧集）使纯片名匹配产生错配（如《贴身保镖》1992 vs 2024）。
方法：
  豆瓣侧：Movie_Name → 维表按片名候选 → 拉丁原名 ⊂ 又名 消歧 → (movie_id, 年份)
  Letterboxd 侧：pkchwy 自带年份，按 (规范化片名, 年份) 分组
  配对：年份差 ≤1 才算同一影片
输出:
  analysis_sets/reviews_samefilms_v2.parquet     年度校验后的主配对集
  analysis_sets/reviews_samefilms_v2_films.csv   影片对照表
  raw/douban_recent/rematch_report.json          消歧统计
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
    ap.add_argument("--min-per-film", type=int, default=4)
    ap.add_argument("--max-per-film", type=int, default=30)
    ap.add_argument("--min-review-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 豆瓣侧：片名+拉丁原名消歧到 (movie_id, year) ----------
    info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                       usecols=["movie_name", "movie_id", "又名", "上映日期"],
                       dtype={"movie_id": str}, low_memory=False)
    info["movie_id"] = info["movie_id"].str.split(".").str[0]
    info["db_year"] = pd.to_numeric(
        info["上映日期"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    info["n_name"] = info["movie_name"].map(norm)
    info["n_aka"] = info["又名"].fillna("").map(norm)
    by_name = {}
    for _, r in info.iterrows():
        by_name.setdefault(r["n_name"], []).append(
            (r["movie_id"], r["db_year"], r["n_aka"], r["movie_name"]))
    print(f"维表 {len(info):,} 行，唯一片名 {len(by_name):,}")

    rz = pd.read_parquet(BASE / "reviews_zh" / "reviews_zh.parquet",
                         columns=["item", "text", "date", "rating", "meta"])
    rz = rz[rz["text"].str.len() >= args.min_review_len]
    name_stats = rz.groupby("item").size()
    print(f"豆瓣语料片名 {len(name_stats):,} 个（≥{args.min_review_len}字）")

    douban_map = {}  # Movie_Name -> (movie_id, db_year, how)
    n_ambig = n_none = 0
    for name in name_stats.index:
        zp, lp = zh_part(name), None
        zp = zh_part(name)
        lp = latin_part(name, zp)
        cands = by_name.get(norm(zp)) or by_name.get(norm(name)) or []
        if not cands:
            n_none += 1
            continue
        if len(cands) == 1:
            douban_map[name] = (cands[0][0], cands[0][1], "unique")
            continue
        lat = norm(lp)
        hit = [c for c in cands if lat and lat in c[2]]
        if len(hit) == 1:
            douban_map[name] = (hit[0][0], hit[0][1], "aka_disambig")
        elif len(hit) > 1:
            years = {h[1] for h in hit}
            if len(years) == 1:
                douban_map[name] = (hit[0][0], hit[0][1], "aka_multi_sameyear")
            else:
                n_ambig += 1
        else:
            n_ambig += 1
    print(f"豆瓣侧解析: 唯一={sum(1 for v in douban_map.values() if v[2]=='unique')}, "
          f"又名消歧={sum(1 for v in douban_map.values() if v[2].startswith('aka'))}, "
          f"歧义放弃={n_ambig}, 无候选={n_none}")

    # ---------- Letterboxd 侧：按 (片名, 年份) 分组 ----------
    lb = pd.read_parquet(BASE / "reviews_en" / "reviews_en_letterboxd.parquet")
    lb["key"] = lb["channel"].map(norm)
    lb["lb_year"] = pd.to_numeric(
        lb["meta"].str.extract(r'"film_year":\s*"(\d{4})"')[0], errors="coerce")
    if lb["lb_year"].isna().mean() > 0.5:
        lb["lb_year"] = pd.to_numeric(
            lb["item"].str.extract(r"\((\d{4})\)")[0], errors="coerce")

    # ---------- 配对：片名 + 年份 ----------
    pairs = []   # (key, douban_name, movie_id, db_year, lb_year)
    for name, (mid, dby, how) in douban_map.items():
        zp = zh_part(name)
        key = norm(latin_part(name, zp))
        if len(key) < 3:
            continue
        lb_c = lb[lb["key"] == key]
        if len(lb_c) == 0:
            continue
        years = sorted(lb_c["lb_year"].dropna().unique())
        hit = [y for y in years if pd.notna(dby) and abs(y - dby) <= 1]
        if hit:
            pairs.append({"matched_film": key, "douban_name": name, "movie_id": mid,
                          "db_year": int(dby) if pd.notna(dby) else None,
                          "lb_year": int(hit[0]), "how": how,
                          "n_zh_pool": int(name_stats[name]),
                          "n_west_pool": int((lb_c["lb_year"] == hit[0]).sum())})
        else:
            pairs.append({"matched_film": key, "douban_name": name, "movie_id": mid,
                          "db_year": int(dby) if pd.notna(dby) else None,
                          "lb_year": None, "how": "year_mismatch",
                          "n_zh_pool": int(name_stats[name]), "n_west_pool": len(lb_c)})
    pr = pd.DataFrame(pairs)
    ok = pr[pr["lb_year"].notna()]
    print(f"\n配对结果: 年度一致 {len(ok):,} 部 / 年度冲突 {(pr.lb_year.isna()).sum():,} 部")
    pr.to_csv(BASE / "raw" / "douban_recent" / "rematch_pairs.csv",
              index=False, encoding="utf-8-sig")

    # ---------- 抽样建集 ----------
    parts = []
    films_rows = []
    for _, p in ok.iterrows():
        zs = rz[rz["item"] == p["douban_name"]]
        ls = lb[(lb["key"] == p["matched_film"]) & (lb["lb_year"] == p["lb_year"])]
        if len(zs) < args.min_per_film or len(ls) < args.min_per_film:
            continue
        zs = zs.sample(min(args.max_per_film, len(zs)), random_state=args.seed)
        ls = ls.sample(min(args.max_per_film, len(ls)), random_state=args.seed)
        n = min(len(zs), len(ls))
        za, la = zs.head(n), ls.head(n)
        parts.append(pd.DataFrame({
            "corpus_id": ["dbv_" + str(p["movie_id"]) + "_" + str(i) for i in range(n)],
            "culture": "zh", "genre": "review", "source": "douban",
            "channel": p["douban_name"], "item": p["douban_name"],
            "date": za["date"].values, "year": pd.to_datetime(za["date"], errors="coerce").dt.year.values,
            "rating": za["rating"].astype(float).values, "sentiment": None, "title": None,
            "text": za["text"].values, "meta": za["meta"].values,
            "matched_film": p["matched_film"]}))
        parts.append(pd.DataFrame({
            "corpus_id": la["corpus_id"].values,
            "culture": "west", "genre": "review", "source": "letterboxd",
            "channel": la["channel"].values, "item": la["item"].values,
            "date": None, "year": None, "rating": None, "sentiment": None,
            "title": None, "text": la["text"].values, "meta": la["meta"].values,
            "matched_film": p["matched_film"]}))
        films_rows.append({"matched_film": p["matched_film"], "douban_name": p["douban_name"],
                           "movie_id": p["movie_id"], "db_year": p["db_year"],
                           "lb_year": p["lb_year"], "n_zh": n, "n_west": n})
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(BASE / "analysis_sets" / "reviews_samefilms_v3.parquet", index=False)
    pd.DataFrame(films_rows).to_csv(
        BASE / "analysis_sets" / "reviews_samefilms_v3_films.csv",
        index=False, encoding="utf-8-sig")
    print(f"\n-> reviews_samefilms_v3.parquet  rows={len(res):,}  "
          f"zh={sum(res.culture=='zh'):,} west={sum(res.culture=='west'):,} "
          f"影片={res['matched_film'].nunique()}")
    rep = {"verified_pairs": len(ok), "year_conflicts": int(pr.lb_year.isna().sum()),
           "douban_ambiguous_dropped": n_ambig, "douban_no_candidate": n_none,
           "final_films": int(res["matched_film"].nunique()), "final_rows": len(res)}
    json.dump(rep, open(BASE / "raw" / "douban_recent" / "rematch_report.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
