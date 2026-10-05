# -*- coding: utf-8 -*-
"""v2：用 id_map.csv 离线解析结果抓取豆瓣近期影评（只调 interests 端点）。

断点续跑：已存在于 reviews_recent.jsonl 的影片自动跳过。
用法: python fetch_douban_recent_v2.py [--since 2023-01-01] [--limit 0]
"""
import argparse
import io
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT_DIR = BASE / "raw" / "douban_recent"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 Version/15.0 Mobile/15E148 Safari/604.1",
      "Accept": "application/json, text/plain, */*",
      "Accept-Language": "zh-CN,zh;q=0.9"}


def http_get(url, referer, retries=3):
    h = dict(UA)
    h["Referer"] = referer
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=h)
            return urllib.request.urlopen(req, timeout=25).read().decode("utf-8")
        except Exception as e:
            if k < retries:
                wait = 25 * (k + 1) + random.uniform(0, 10)
                print(f"    请求失败 {e}，{wait:.0f}s 后重试", flush=True)
                time.sleep(wait)
            else:
                raise


def fetch_recent(fid, since, max_pages):
    rows, seen = [], set()
    for start in range(0, max_pages * 20, 20):
        u = (f"https://m.douban.com/rexxar/api/v2/movie/{fid}/interests"
             f"?count=20&start={start}&order_by=time")
        d = json.loads(http_get(u, f"https://m.douban.com/movie/subject/{fid}/comments"))
        ints = d.get("interests", [])
        if not ints:
            break
        for r in ints:
            rid = str(r.get("id"))
            dt = (r.get("create_time") or "")[:10]
            if rid in seen or dt < since:
                continue
            seen.add(rid)
            rows.append({
                "review_id": rid, "date": dt,
                "rating": (r.get("rating") or {}).get("value"),
                "text": " ".join((r.get("comment") or "").split()),
                "vote_count": r.get("vote_count", 0),
                "ip_location": r.get("ip_location") or ""})
        time.sleep(random.uniform(1.6, 2.6))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="", help="输出文件名后缀（如 _2020plus 独立窗口）")
    ap.add_argument("--force", action="store_true", help="忽略已有记录全量重抓")
    args = ap.parse_args()

    idmap = pd.read_csv(OUT_DIR / "id_map.csv", dtype={"movie_id": str})
    idmap["movie_id"] = idmap["movie_id"].map(
        lambda x: x.split(".")[0] if isinstance(x, str) else "")
    idmap = idmap[idmap["movie_id"].str.len() > 0]
    if args.limit:
        idmap = idmap.head(args.limit)

    out_f = OUT_DIR / f"reviews_recent{args.suffix}.jsonl"
    already = set()
    if out_f.exists() and not args.force:
        for l in open(out_f, encoding="utf-8"):
            if l.strip():
                already.add(json.loads(l)["douban_name"])
    print(f"维表解析 {len(idmap)} 部；已抓取 {len(already)} 部，跳过")

    n_done, n_rev = 0, 0
    with open(out_f, "a", encoding="utf-8") as fout:
        for _, film in idmap.iterrows():
            name = film["douban_name"]
            if name in already:
                continue
            fid = str(film["movie_id"])
            try:
                rows = fetch_recent(fid, args.since, args.max_pages)
            except Exception as e:
                print(f"[抓取失败] {name} ({fid}): {e}", flush=True)
                continue
            for r in rows:
                r.update({"film_id": fid, "film_title": film["zh_title"],
                          "douban_name": name, "matched_film": film["matched_film"]})
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            n_rev += len(rows)
            if n_done % 10 == 0 or len(rows) >= 15:
                print(f"[{n_done}/{len(idmap) - len(already)}] {name}({fid}): "
                      f"近期评论 {len(rows)}（累计 {n_rev}）", flush=True)
            time.sleep(random.uniform(1.0, 1.8))
    print(f"\n完成 {n_done} 部，共新增 {n_rev} 条 -> {out_f}")


if __name__ == "__main__":
    main()
