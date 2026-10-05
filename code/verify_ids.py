# -*- coding: utf-8 -*-
"""验证慢速解析的影片 ID：rexxar movie/{id} 元信息年份 vs Letterboxd 年份。"""
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "douban_recent"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 Version/15.0 Mobile/15E148 Safari/604.1",
      "Accept": "application/json, text/plain, */*",
      "Referer": "https://m.douban.com/"}


def http_get(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=20).read().decode("utf-8")


# Letterboxd 年份（films 表 letterboxd_item = 'Title (year)'）
films = pd.read_csv(BASE / "analysis_sets" / "reviews_samefilms_films.csv")
films["lb_year"] = films["letterboxd_item"].str.extract(r"\((\d{4})\)")[0]
idmap = pd.read_csv(OUT / "id_map.csv", dtype=str).fillna("")
chk = idmap[idmap["how"].isin(["rexxar_slow", "aka_latin"])]
print(f"待验证 {len(chk)} 部（rexxar_slow + aka_latin）")

rows = []
for _, f in chk.iterrows():
    fid = f["movie_id"]
    rec = {"matched_film": f["matched_film"], "douban_name": f["douban_name"],
           "movie_id": fid, "how": f["how"]}
    try:
        d = json.loads(http_get(f"https://m.douban.com/rexxar/api/v2/movie/{fid}"))
        rec["db_title"] = d.get("title", "")
        rec["db_year"] = d.get("year", "")
        rec["db_id"] = str(d.get("id", ""))
    except Exception as e:
        rec["db_title"] = f"ERR {e}"
    rows.append(rec)
    print(f"  {f['douban_name']} [{fid}] -> {rec.get('db_title')} ({rec.get('db_year')})", flush=True)
    time.sleep(2.5)

res = pd.DataFrame(rows).merge(
    films[["matched_film", "letterboxd_item", "lb_year"]], on="matched_film", how="left")
res["year_diff"] = (pd.to_numeric(res["db_year"], errors="coerce") -
                    pd.to_numeric(res["lb_year"], errors="coerce")).abs()
res["verdict"] = "ok"
res.loc[res["year_diff"] > 1, "verdict"] = "wrong_year"
res.loc[res["db_title"].astype(str).str.startswith("ERR"), "verdict"] = "fetch_fail"
res.to_csv(OUT / "id_verify.csv", index=False, encoding="utf-8-sig")
print("\n判定分布:", res["verdict"].value_counts().to_dict())
print("\n可疑条目（year_diff>1 或获取失败）:")
print(res[res.verdict != "ok"][["douban_name", "movie_id", "db_title", "db_year", "lb_year"]].to_string(index=False))
