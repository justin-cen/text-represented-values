# -*- coding: utf-8 -*-
"""离线解析 717 部配对影片的豆瓣 subject_id（用 douban_movie_info 维表）。

输出: data_corpus/raw/douban_recent/id_map.csv
匹配策略（按序回退）:
  1) 中文片名精确匹配 movie_name
  2) 规范化后精确匹配（去空格/标点/大小写）
  3) 拉丁片名 ⊂ 又名 字段（原名/译名）
"""
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "douban_recent"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CJK_PREFIX = re.compile(r"^([\u4e00-\u9fff：·、，。《》！？（）\s]+?)\s*[A-Za-z]")


def zh_title(name):
    m = CJK_PREFIX.match(name or "")
    return (m.group(1).strip() if m else (name or "").strip())


def norm(s):
    s = re.sub(r"[\s：:·、，。《》！？（）()\-—–'\".,!?&/]+", "", str(s))
    return s.lower()


def main():
    films = pd.read_csv(BASE / "analysis_sets" / "reviews_samefilms_films.csv")
    info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                       usecols=["movie_name", "movie_id", "又名", "IMDb", "score", "上映日期"],
                       low_memory=False)
    print(f"维表影片数: {len(info):,}")
    info["movie_id"] = info["movie_id"].astype(str)
    info["norm_name"] = info["movie_name"].map(norm)
    by_exact = dict(zip(info["movie_name"], info["movie_id"]))
    by_norm = {}
    for _, r in info.iterrows():
        by_norm.setdefault(r["norm_name"], r["movie_id"])
    aka = [(norm(str(a)) if pd.notna(a) else "", mid) for a, mid in
           zip(info["又名"], info["movie_id"])]

    rows, n_hit = [], 0
    for _, f in films.iterrows():
        name, key = f["douban_name"], f["matched_film"]
        zt = zh_title(name)
        latin = name[len(zt):].strip() if zt else name
        mid, how = None, ""
        if zt in by_exact:
            mid, how = by_exact[zt], "exact_cn"
        elif norm(zt) in by_norm:
            mid, how = by_norm[norm(zt)], "norm_cn"
        elif norm(name) in by_norm:
            mid, how = by_norm[norm(name)], "norm_full"
        else:
            lat = norm(latin)
            if len(lat) >= 4:
                cands = [m for a, m in aka if lat in a]
                if cands:
                    mid, how = cands[0], "aka_latin"
        if mid:
            n_hit += 1
        rows.append({"matched_film": key, "douban_name": name, "zh_title": zt,
                     "movie_id": mid or "", "how": how})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "id_map.csv", index=False, encoding="utf-8-sig")
    print(f"解析成功: {n_hit}/{len(films)}")
    print(out["how"].value_counts().to_dict())
    miss = out[out["movie_id"] == ""]
    if len(miss):
        print("未命中示例:", miss["douban_name"].head(10).tolist())


if __name__ == "__main__":
    main()
