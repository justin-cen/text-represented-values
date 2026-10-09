# -*- coding: utf-8 -*-
"""省级新闻统一采集 v2：人民网省级子域（31 省全覆盖）。

流程：Playwright 渲染「首页 + 要闻列表页」→ 收集文章 URL → 静态抓正文 → 归省存档。
带断点续跑与限流延迟。
输出: data_corpus/raw/province_news/pd_news.jsonl + pd_news.parquet
用法: python scrape_province_news_pd.py [--per-province 60] [--delay 0.6]
"""
import argparse
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus\raw\province_news")
BASE.mkdir(parents=True, exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
SUBS = json.loads(Path(r"E:\culture-difference\work\pd_subdomains.json").read_text(encoding="utf-8"))
ART_PAT = re.compile(r"/n2/\d+/\d+/c\d+-\d+\.html")


def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_text(html):
    import trafilatura
    return " ".join((trafilatura.extract(html) or "").split())


def render_urls(pg, url, sub):
    """渲染页面并抽取该子域的文章 URL。"""
    try:
        pg.goto(url, timeout=35000, wait_until="domcontentloaded")
        pg.wait_for_timeout(2200)
        html = pg.content()
    except Exception:
        return []
    found = set(re.findall(r"https?://" + sub + r"\.people\.com\.cn" + ART_PAT.pattern, html))
    try:
        links = pg.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        found |= set(u for u in links if sub + ".people.com.cn" in u and ART_PAT.search(u))
    except Exception:
        pass
    return sorted(found)


def find_yaowen(pg, home, sub):
    """在首页找“要闻”栏目列表页 URL。"""
    try:
        pg.goto(home, timeout=35000, wait_until="domcontentloaded")
        pg.wait_for_timeout(2000)
        links = pg.eval_on_selector_all(
            "a[href]", "els => els.map(e => [e.href, (e.textContent||'').trim()])")
    except Exception:
        return None
    for u, t in links:
        if sub + ".people.com.cn" in u and t in ("要闻", "要闻动态", "今日要闻", "本网关注"):
            return u
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-province", type=int, default=60)
    ap.add_argument("--delay", type=float, default=0.6)
    args = ap.parse_args()
    out_f = BASE / "pd_news.jsonl"

    done_prov = set()
    if out_f.exists():
        for l in open(out_f, encoding="utf-8"):
            if l.strip():
                try:
                    done_prov.add(json.loads(l)["province"])
                except Exception:
                    pass
        print(f"断点续跑：已完成 {len(done_prov)} 省: {sorted(done_prov)}")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(user_agent=UA["User-Agent"])
        with open(out_f, "a", encoding="utf-8") as fw:
            for prov, sub in SUBS.items():
                if prov in done_prov:
                    continue
                home = f"http://{sub}.people.com.cn/"
                urls = render_urls(pg, home, sub)
                yw = find_yaowen(pg, home, sub)
                if yw:
                    urls = sorted(set(urls) | set(render_urls(pg, yw, sub)))
                n, seen = 0, set()
                for u in urls:
                    if n >= args.per_province:
                        break
                    if u in seen:
                        continue
                    seen.add(u)
                    try:
                        html = http_get(u)
                        txt = extract_text(html)
                        if len(txt) < 200:
                            continue
                        title = re.search(r"<title>([^<]+)</title>", html)
                        title = title.group(1).strip() if title else ""
                        m = re.search(r"/n2/(\d+)/(\d+)/", u)
                        ymd = f"{m.group(1)}-{m.group(2)[:2]}-{m.group(2)[2:]}" if m else ""
                        rec = {"corpus_id": "pdprov_" + u.split("/")[-1].replace(".html", ""),
                               "province": prov, "culture": "zh", "genre": "news",
                               "source": "people_province", "channel": prov, "item": title,
                               "date": ymd, "year": int(m.group(1)) if m else 2026,
                               "title": title, "text": txt,
                               "meta": json.dumps({"url": u}, ensure_ascii=False)}
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                    except Exception:
                        continue
                    time.sleep(args.delay)
                print(f"[{prov}/{sub}] {n} 篇（候选 {len(urls)}，要闻页 {'有' if yw else '无'}）", flush=True)
                time.sleep(2)
        b.close()

    recs = [json.loads(l) for l in open(out_f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(recs).drop_duplicates("corpus_id")
    df.to_parquet(BASE / "pd_news.parquet", index=False)
    print(f"\n完成：{len(df):,} 篇 / {df.province.nunique()} 省")
    print(df.groupby("province").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
