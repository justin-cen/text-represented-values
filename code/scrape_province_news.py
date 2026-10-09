# -*- coding: utf-8 -*-
"""省级新闻统一采集：中新网省级子域（~24 省，静态 .shtml，GBK 编码）。

每省抓子域首页文章链接 → 抓正文 → 归省存档。
输出 data_corpus/raw/province_news/province_news.jsonl + province_news.parquet
用法: python scrape_province_news.py [--per-province 50] [--delay 0.35]
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

# 中新网省级子域（经探测可用）
SUBS = {
    "北京": "bj", "天津": "tj", "重庆": "cq", "上海": "sh", "广东": "gd", "广西": "gx",
    "福建": "fj", "江西": "jx", "湖南": "hn", "湖北": "hb", "河南": "henan",
    "浙江": "zj", "江苏": "js", "山东": "sd", "四川": "sc", "云南": "yn", "贵州": "gz",
    "陕西": "sn", "山西": "sx", "河北": "he", "辽宁": "ln", "吉林": "jl", "黑龙江": "hl",
    "甘肃": "gs", "青海": "qh", "宁夏": "nx", "新疆": "xj", "内蒙古": "nm", "安徽": "ah",
    "海南": "hi",
}
# 补充 js/sd/he/hl/nm/hi 的备用域名（部分省用拼音全称）
ALT = {"江苏": "jiangsu", "山东": "shandong", "河北": "hebei", "黑龙江": "heilongjiang",
       "内蒙古": "neimenggu", "海南": "hainan"}


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


def province_links(sub_home, prov):
    html = http_get(sub_home)
    # 全部 .shtml 链接（含省级子域与全国域名；省级子域的内容聚合即视为省域来源）
    links = re.findall(r'href="(https?://[^"]+\.shtml)"', html)
    links = [u for u in dict.fromkeys(links) if "/20" in u]
    return links


def is_provincial(txt, title, prov):
    """判省域归属：来源行含省名，或正文/标题含省名或其省会/主要城市名 ≥2 次。"""
    s = (txt + " " + title)
    if f"中新网{prov}新闻" in s or f"{prov}网" in s or f"中新网{prov}" in s:
        return True
    # 内容中省名出现次数
    return s.count(prov) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-province", type=int, default=50)
    ap.add_argument("--delay", type=float, default=0.35)
    ap.add_argument("--provinces", default="", help="逗号分隔省名；空为全部")
    args = ap.parse_args()
    out_f = BASE / "province_news.jsonl"

    done_prov = set()
    if out_f.exists():
        for l in open(out_f, encoding="utf-8"):
            if l.strip():
                try:
                    done_prov.add(json.loads(l)["province"])
                except Exception:
                    pass
        print(f"断点续跑：已有省份 {len(done_prov)} 个: {sorted(done_prov)}")

    targets = [p for p in SUBS if (not args.provinces or p in args.provinces)]
    targets = [p for p in targets if p not in done_prov]

    with open(out_f, "a", encoding="utf-8") as fw:
        for prov in targets:
            subs = [SUBS[prov]]
            if prov in ALT:
                subs = [ALT[prov], SUBS[prov]]
            links = []
            for s in subs:
                try:
                    home = f"https://www.{s}.chinanews.com.cn/"
                    links = province_links(home, prov)
                    if links:
                        break
                except Exception as e:
                    print(f"  [{prov}] {s} 首页失败: {type(e).__name__}", flush=True)
            if not links:
                print(f"[{prov}] 无链接，跳过", flush=True)
                continue
            take = links[: args.per_province]
            n = 0
            for u in take:
                try:
                    html = http_get(u)
                    txt = extract_text(html)
                    title = re.search(r"<title>([^<]+)</title>", html)
                    title = title.group(1).strip() if title else ""
                    if len(txt) < 200 or not is_provincial(txt, title, prov):
                        continue
                    date_m = re.search(r"(20\d\d)-(\d\d)-(\d\d)", u)
                    ymd = date_m.group(0) if date_m else ""
                    rec = {"corpus_id": "provnews_" + u.split("/")[-1].replace(".shtml", ""),
                           "province": prov, "culture": "zh", "genre": "news",
                           "source": "chinanews_province", "channel": prov, "item": title,
                           "date": ymd, "year": int(ymd[:4]) if ymd else 2026,
                           "title": title, "text": txt,
                           "meta": json.dumps({"url": u}, ensure_ascii=False)}
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                except Exception:
                    continue
                time.sleep(args.delay)
            print(f"[{prov}] {n} 篇（链接 {len(links)}）", flush=True)

    recs = [json.loads(l) for l in open(out_f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(recs).drop_duplicates("corpus_id")
    df.to_parquet(BASE / "province_news.parquet", index=False)
    print(f"\n完成：{len(df):,} 篇 / {df.province.nunique()} 省")
    print("各省篇数:")
    print(df.groupby("province").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
