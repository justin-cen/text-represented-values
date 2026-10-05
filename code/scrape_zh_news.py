# -*- coding: utf-8 -*-
"""抓取 2026 年 8-9 月中国主流媒体近期新闻（直连官方站点，礼貌限速）。

来源：
  rmrb      人民日报电子版 paper.people.com.cn（可回溯任意日期，完整版面）
  xinhua    新华网频道页 www.news.cn（仅近数日，作为补充）
  chinadaily 中国日报 chinadaily.com.cn（英文，分页可回溯）

输出: raw/scrape_2026/{outlet}.parquet  (url, outlet, culture, date, title, text, page)

用法:
  python scrape_zh_news.py --outlet rmrb --date-min 2026-08-01 --date-max 2026-09-27
  python scrape_zh_news.py --outlet xinhua
  python scrape_zh_news.py --outlet chinadaily --pages 14
"""
import argparse
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pandas as pd
import trafilatura

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "scrape_2026"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"}
CLIENT = httpx.Client(headers=UA, timeout=30, follow_redirects=True)


def fetch_html(url: str) -> str | None:
    try:
        r = CLIENT.get(url)
        if r.status_code != 200:
            return None
        r.encoding = r.charset_encoding or "utf-8"
        return r.text
    except Exception:
        return None


def extract(url: str, outlet: str, culture: str, min_len: int = 300):
    html = fetch_html(url)
    if not html:
        return None
    try:
        res = trafilatura.extract(html, output_format="json", with_metadata=True,
                                  include_comments=False, favor_recall=True)
        if not res:
            return None
        m = json.loads(res)
        text = " ".join((m.get("text") or "").split())
        if len(text) < min_len:
            return None
        return {"url": url, "outlet": outlet, "culture": culture,
                "date": (m.get("date") or "")[:10] or None,
                "title": (m.get("title") or "").strip() or None, "text": text}
    except Exception:
        return None


def harvest(article_urls, outlet, culture, workers=5, delay=0.25):
    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract, u, outlet, culture): u for u in article_urls}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r:
                rows.append(r)
            if done % 100 == 0:
                print(f"  正文进度 {done}/{len(article_urls)} 成功 {len(rows)}")
            time.sleep(delay / workers)
    return rows


# ---------------------------------------------------------------- 人民日报
def collect_rmrb(d0: date, d1: date) -> list[str]:
    urls, seen = [], set()
    day = d0
    while day <= d1:
        ym, ymd = day.strftime("%Y%m"), day.strftime("%Y%m/%d")
        for node in range(1, 26):
            u = f"http://paper.people.com.cn/rmrb/pc/layout/{ymd}/node_{node:02d}.html"
            html = fetch_html(u)
            if not html:
                break
            links = re.findall(r'href="([^"]*content/2026\d+/\d+/content_\d+\.html)"', html)
            for l in links:
                full = urljoin(u, l)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        if day.day in (1, 10, 20):
            print(f"  人民日报已收集到 {day}: 累计 {len(urls)}")
        day += timedelta(days=1)
        time.sleep(0.2)
    return urls


# ---------------------------------------------------------------- 新华网
XINHUA_CHANNELS = ["politics", "world", "fortune", "tech", "edu", "ent", "legal",
                   "local", "comments", "society"]


def collect_xinhua() -> list[str]:
    urls, seen = [], set()
    pat = re.compile(r"https?://www\.news\.cn/(?:[a-z]+/)?2026\d{4}/[0-9a-f]{32}/c\.html")
    for ch in XINHUA_CHANNELS:
        for u in (f"https://www.news.cn/{ch}/", f"https://www.news.cn/{ch}/index.htm"):
            html = fetch_html(u)
            if not html:
                continue
            for l in re.findall(r'href="([^"]+)"', html):
                full = urljoin(u, l)
                if pat.match(full) and full not in seen:
                    seen.add(full)
                    urls.append(full)
        print(f"  新华网[{ch}]: 累计 {len(urls)}")
        time.sleep(0.3)
    return urls


# ---------------------------------------------------------------- 中国日报
CD_SECTIONS = ["china/governmentandpolicy", "china/society", "china/trending",
               "world", "business", "business/tech", "culture/lifestyle",
               "opinion", "sports"]


def collect_chinadaily(max_pages: int, date_min: str) -> list[str]:
    urls, seen = [], set()
    pat = re.compile(r'//www\.chinadaily\.com\.cn/a/(2026\d{2})/(\d{2})/[^"]+?\.html')
    for sec in CD_SECTIONS:
        for page in range(1, max_pages + 1):
            u = (f"https://www.chinadaily.com.cn/{sec}" if page == 1
                 else f"https://www.chinadaily.com.cn/{sec}/page_{page}.html")
            html = fetch_html(u)
            if not html:
                break
            links = re.findall(r'href="([^"]+?\.html)"', html)
            n_new, oldest = 0, None
            for l in links:
                m = pat.search(l)
                if not m:
                    continue
                d = f"{m.group(1)[:4]}-{m.group(1)[4:]}-{m.group(2)}"
                oldest = d if not oldest else min(oldest, d)
                full = "https:" + l if l.startswith("//") else l
                if d >= date_min and full not in seen:
                    seen.add(full)
                    urls.append(full)
                    n_new += 1
            print(f"  CD[{sec}] page{page}: 新增 {n_new} 累计 {len(urls)} 最旧 {oldest}")
            time.sleep(0.3)
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlet", required=True, choices=["rmrb", "xinhua", "chinadaily"])
    ap.add_argument("--date-min", default="2026-08-01")
    ap.add_argument("--date-max", default="2026-09-27")
    ap.add_argument("--pages", type=int, default=14)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    if args.outlet == "rmrb":
        d0 = date.fromisoformat(args.date_min)
        d1 = date.fromisoformat(args.date_max)
        urls = collect_rmrb(d0, d1)
        culture = "zh"
    elif args.outlet == "xinhua":
        urls = collect_xinhua()
        culture = "zh"
    else:
        urls = collect_chinadaily(args.pages, args.date_min)
        culture = "zh_en"

    print(f"{args.outlet}: 收集到 {len(urls)} 个文章 URL，开始抓正文…")
    rows = harvest(urls, args.outlet, culture, workers=args.workers)
    df = pd.DataFrame(rows)
    if len(df):
        if args.outlet == "chinadaily":
            df = df[df["date"] >= args.date_min]
        df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["text"])
    out = OUT / f"{args.outlet}.parquet"
    df.to_parquet(out, index=False)
    print(f"== {args.outlet}: 保存 {len(df):,} 篇 -> {out}")
    if len(df):
        print("   日期范围:", df["date"].min(), "→", df["date"].max())


if __name__ == "__main__":
    main()
