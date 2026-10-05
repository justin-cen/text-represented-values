# -*- coding: utf-8 -*-
"""抓取中新网/央视/央广网 首页+频道页近期文章（多样性补充，仅近数日）。

输出: raw/scrape_2026/{portal}.parquet (url, outlet, culture, date, title, text)
日期直接从 URL 解析（不依赖 trafilatura 元数据）。

用法: python scrape_portals.py --portal chinanews
      python scrape_portals.py --portal cctv
      python scrape_portals.py --portal cnr
"""
import argparse
import io
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).parent))
from scrape_zh_news import harvest  # noqa: E402

BASE = Path(r"E:\culture-difference\data_corpus")
OUT = BASE / "raw" / "scrape_2026"
OUT.mkdir(parents=True, exist_ok=True)
# 注意：不要再包装 sys.stdout —— scrape_zh_news 导入时已包装，
# 重复包装会导致底层 buffer 被关闭（I/O operation on closed file）。

from scrape_zh_news import fetch_html  # noqa: E402

PORTALS = {
    "chinanews": {
        "seeds": ["https://www.chinanews.com.cn/", "https://www.chinanews.com.cn/cj/",
                  "https://www.chinanews.com.cn/sh/", "https://www.chinanews.com.cn/gj/",
                  "https://www.chinanews.com.cn/cul/", "https://www.chinanews.com.cn/kj/",
                  "https://www.chinanews.com.cn/yl/", "https://www.chinanews.com.cn/gn/"],
        "pat": re.compile(r"//www\.chinanews\.com\.cn/(?:[a-z]+/)*?(2026)/(\d{2})-(\d{2})/\d+\.shtml"),
        "outlet": "中新网",
    },
    "cctv": {
        "seeds": ["https://news.cctv.com/", "https://news.cctv.com/china/",
                  "https://news.cctv.com/world/", "https://jingji.cctv.com/",
                  "https://news.cctv.com/society/", "https://tech.cctv.com/"],
        "pat": re.compile(r"https?://(?!tv\.)[a-z]+\.cctv\.com/(2026)/(\d{2})/(\d{2})/ARTI[^\"/]+\.shtml"),
        "outlet": "央视网",
    },
    "cnr": {
        "seeds": ["https://www.cnr.cn/", "https://news.cnr.cn/native/gd/",
                  "https://news.cnr.cn/native/city/", "https://finance.cnr.cn/",
                  "https://tech.cnr.cn/", "https://ent.cnr.cn/"],
        "pat": re.compile(r"https?://[a-z]+\.cnr\.cn/[^\"]*?/t(2026)(\d{2})(\d{2})_\d+\.shtml"),
        "outlet": "央广网",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portal", required=True, choices=list(PORTALS))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    cfg = PORTALS[args.portal]

    urls, seen = [], set()
    for seed in cfg["seeds"]:
        html = fetch_html(seed)
        if not html:
            print(f"  [{args.portal}] {seed} 获取失败")
            continue
        for l in re.findall(r'href="([^"]+)"', html) + re.findall(r"href='([^']+)'", html):
            full = urljoin(seed, l if not l.startswith("//") else "https:" + l)
            m = cfg["pat"].search(full)
            if m and full not in seen:
                seen.add(full)
                urls.append(full)
        print(f"  [{args.portal}] {seed}: 累计 {len(urls)}")
        time.sleep(0.3)

    print(f"{args.portal}: 收集 {len(urls)} 个 URL，抓正文…")
    rows = harvest(urls, cfg["outlet"], "zh", workers=args.workers)
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df):
        # 用 URL 日期覆盖
        def ud(u):
            m = cfg["pat"].search(u)
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        df["date"] = df["url"].map(ud)
        df = df.dropna(subset=["date"]).drop_duplicates(subset=["url"]).drop_duplicates(subset=["text"])
    out = OUT / f"{args.portal}.parquet"
    df.to_parquet(out, index=False)
    print(f"== {args.portal}: 保存 {len(df):,} 篇 -> {out}")
    if len(df):
        print("   日期:", df["date"].min(), "→", df["date"].max())


if __name__ == "__main__":
    main()
