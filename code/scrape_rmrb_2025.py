# -*- coding: utf-8 -*-
"""抓取人民日报 2025 年电子版（paper.people.com.cn/rmrb），用于与 FineNews 2025 同年对齐。

输出: data_corpus/raw/scrape_2026/rmrb_2025.parquet
用法: python scrape_rmrb_2025.py [--start 2025-01-01] [--end 2025-09-30] [--max-per-day 12]
"""
import argparse
import io
import re
import sys
import time
import urllib.request
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "scrape_2026"
OUT.mkdir(exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
      "Accept-Language": "zh-CN,zh;q=0.9"}
HOST = "http://paper.people.com.cn"


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")


def day_links(y, m, d):
    """某日全部版面页的文章链接。"""
    base = f"{HOST}/rmrb/pc/layout/{y}{m:02d}/{d:02d}"
    arts, seen_pages = [], 0
    for nn in range(1, 30):
        try:
            html = http_get(f"{base}/node_{nn:02d}.html")
        except Exception:
            break
        if "您访问的页面不存在" in html or len(html) < 3000:
            break
        seen_pages += 1
        for href in re.findall(r'href="([^"]*content/\d{6}/\d{2}/content_\d+\.html)"', html):
            u = urllib.parse.urljoin(f"{base}/", href)
            if u not in arts:
                arts.append(u)
    return arts


def fetch_article(url, ymd):
    import trafilatura
    try:
        html = http_get(url)
        txt = trafilatura.extract(html) or ""
        txt = " ".join(txt.split())
        if len(txt) < 100:
            return None
        tm = re.search(r"<title>([^<]+)</title>", html)
        title = tm.group(1).strip() if tm else ""
        return {"corpus_id": "rmrb25_" + url.split("content_")[-1].replace(".html", ""),
                "culture": "zh", "genre": "news", "source": "rmrb", "channel": "人民日报",
                "item": "人民日报", "date": ymd, "year": 2025, "rating": None,
                "sentiment": None, "title": title, "text": txt,
                "meta": '{"url": "%s"}' % url}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--max-per-day", type=int, default=12)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 _full")
    args = ap.parse_args()
    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    rows, n_day = [], 0
    cur = d0
    out_f = OUT / f"rmrb_2025{args.tag}.jsonl"
    done = {}
    if out_f.exists():
        for l in open(out_f, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["date"]] = done.get(r["date"], 0) + 1
        print(f"断点续跑：已完成 {len(done)} 天；已达配额的天数 "
              f"{sum(1 for v in done.values() if v >= args.max_per_day)}")
    import json
    with open(out_f, "a", encoding="utf-8") as fw:
        while cur <= d1:
            ymd = cur.isoformat()
            if done.get(ymd, 0) >= args.max_per_day:      # 该日已达配额 → 跳过
                cur += timedelta(days=1); continue
            y, m, d = cur.year, cur.month, cur.day
            try:
                links = day_links(y, m, d)
            except Exception as e:
                print(f"  {ymd} 版面索引失败: {e}", flush=True)
                cur += timedelta(days=1); continue
            need = args.max_per_day - done.get(ymd, 0)
            got = 0
            for u in links[done.get(ymd, 0): done.get(ymd, 0) + need]:
                rec = fetch_article(u, ymd)
                if rec:
                    rows.append(rec); got += 1
                time.sleep(args.delay)
            for rec in rows:
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.clear()
            n_day += 1
            if n_day % 20 == 0:
                print(f"  已处理 {n_day} 天（{ymd}），当日新增 {got} 篇", flush=True)
            time.sleep(args.delay)
            cur += timedelta(days=1)
    # 转 parquet
    recs = [json.loads(l) for l in open(out_f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(recs).drop_duplicates("corpus_id")
    df.to_parquet(OUT / f"rmrb_2025{args.tag}.parquet", index=False)
    print(f"完成：{len(df):,} 篇 -> {OUT / f'rmrb_2025{args.tag}.parquet'}")
    print("月份分布:", df["date"].str[:7].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
