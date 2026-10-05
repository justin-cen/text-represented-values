# -*- coding: utf-8 -*-
"""主配对集（717 部）年度一致性审计 + id_map 清理。

douban 侧年份：id_map movie_id → 维表 上映日期 首个 4 位年份
letterboxd 侧年份：films 表 letterboxd_item 括号年份
输出: raw/douban_recent/year_audit.csv + id_map_clean.csv（剔除年度不符且非精确匹配的）
"""
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "douban_recent"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

films = pd.read_csv(BASE / "analysis_sets" / "reviews_samefilms_films.csv")
films["lb_year"] = pd.to_numeric(
    films["letterboxd_item"].str.extract(r"\((\d{4})\)")[0], errors="coerce")
idmap = pd.read_csv(OUT / "id_map.csv", dtype=str).fillna("")
info = pd.read_csv(BASE / "raw" / "douban_info" / "douban_movie_info.csv",
                   usecols=["movie_id", "上映日期", "movie_name"], dtype={"movie_id": str},
                   low_memory=False)
info["movie_id"] = info["movie_id"].str.split(".").str[0]
info["db_year"] = pd.to_numeric(
    info["上映日期"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
# 同名电影多条：保留首个（维表内 Typerank 排序，首个通常是热度最高/最经典版本）
info = info.drop_duplicates("movie_id")
y_by_id = dict(zip(info["movie_id"], info["db_year"]))

aud = films.merge(idmap[["matched_film", "movie_id", "how"]], on="matched_film", how="left")
aud["db_year"] = aud["movie_id"].map(y_by_id)
aud["year_diff"] = (aud["db_year"] - aud["lb_year"]).abs()

def verdict(r):
    if not r["movie_id"]:
        return "unresolved"
    if pd.isna(r["db_year"]) or pd.isna(r["lb_year"]):
        return "unknown"
    if r["year_diff"] <= 1:
        return "consistent"
    return "mismatch"

aud["verdict"] = aud.apply(verdict, axis=1)
aud.to_csv(OUT / "year_audit.csv", index=False, encoding="utf-8-sig")
print("717 部审计结果:")
print(aud["verdict"].value_counts().to_dict())
print("\n按解析方式:")
print(aud.groupby(["how", "verdict"]).size().unstack(fill_value=0).to_string())
print("\n年度不符（mismatch）示例:")
cols = ["matched_film", "douban_name", "letterboxd_item", "db_year", "lb_year", "how"]
print(aud[aud.verdict == "mismatch"][cols].head(25).to_string(index=False))

# 清理 id_map：mismatch 的搜索/又名解析置空（exact_cn 因片名精确来自豆瓣语料，保留但标记）
idmap = idmap.merge(aud[["matched_film", "verdict", "db_year", "lb_year"]],
                    on="matched_film", how="left")
mask = (idmap["verdict"] == "mismatch") & (idmap["how"].isin(["rexxar_slow", "aka_latin"]))
idmap.loc[mask, "movie_id"] = ""
idmap.loc[mask, "how"] = "dropped_year_mismatch"
idmap.to_csv(OUT / "id_map.csv", index=False, encoding="utf-8-sig")
print(f"\nid_map 清理：剔除 {mask.sum()} 部年度不符的非精确解析")
