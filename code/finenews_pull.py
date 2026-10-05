# -*- coding: utf-8 -*-
"""从 FineNews（CC-News 2021-2025，HF: ksolovev/fine-news）抽取中外主流媒体近期新闻。

策略：对指定月份，英文每月随机抽 n_en 个分片、中文抽 n_zh 个分片下载，
然后按 hostname 白名单过滤主流中外媒体，保存到 raw/finenews_filtered/。

用法: python finenews_pull.py [--months 2025-08,2025-09,2025-10] [--n-en 8] [--n-zh 60]
"""
import argparse
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
RAW = BASE / "raw" / "finenews"
FIL = BASE / "raw" / "finenews_filtered"
FIL.mkdir(parents=True, exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API = "https://hf-mirror.com/api/datasets/ksolovev/fine-news/tree/main/{month}/{lang}"
DL = "https://hf-mirror.com/datasets/ksolovev/fine-news/resolve/main/{month}/{lang}/{name}"

# hostname 后缀 -> (outlet, culture)
HOST_MAP = {
    "bbc.com": ("BBC", "west"), "bbc.co.uk": ("BBC", "west"),
    "theguardian.com": ("Guardian", "west"),
    "nytimes.com": ("NYT", "west"),
    "cnn.com": ("CNN", "west"),
    "reuters.com": ("Reuters", "west"),
    "apnews.com": ("AP", "west"),
    "washingtonpost.com": ("WaPo", "west"),
    "foxnews.com": ("Fox", "west"),
    "npr.org": ("NPR", "west"),
    "usatoday.com": ("USAToday", "west"),
    "independent.co.uk": ("Independent", "west"),
    "telegraph.co.uk": ("Telegraph", "west"),
    "news.cn": ("新华社", "zh"), "xinhuanet.com": ("新华社", "zh"),
    "people.com.cn": ("人民网", "zh"),
    "chinadaily.com.cn": ("中国日报", "zh"),
    "cctv.com": ("央视", "zh"),
    "gmwb.cn": ("光明网", "zh"),
    "cnr.cn": ("央广网", "zh"),
    "chinanews.com.cn": ("中新网", "zh"), "chinanews.com": ("中新网", "zh"),
    "thepaper.cn": ("澎湃新闻", "zh"),
    "jiemian.com": ("界面新闻", "zh"),
    "caixin.com": ("财新", "zh"),
    "yicai.com": ("第一财经", "zh"),
}


def host_match(url: str, hostname: str):
    h = (hostname or "").lower()
    if not h and url:
        try:
            h = url.split("/")[2].lower()
        except Exception:
            return None
    for suf, v in HOST_MAP.items():
        if h == suf or h.endswith("." + suf) or h.endswith(suf):
            return v
    return None


def list_files(client, month, lang):
    r = client.get(API.format(month=month, lang=lang), timeout=60)
    r.raise_for_status()
    return [x["path"].split("/")[-1] for x in r.json() if x["type"] == "file"]


def download(client, month, lang, name):
    out = RAW / month / lang / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10000:
        return out
    r = client.get(DL.format(month=month, lang=lang, name=name), timeout=300)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2025-08,2025-09,2025-10")
    ap.add_argument("--n-en", type=int, default=8)
    ap.add_argument("--n-zh", type=int, default=60)
    ap.add_argument("--min-len", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    months = args.months.split(",")
    rng = random.Random(args.seed)
    client = httpx.Client(follow_redirects=True,
                          headers={"User-Agent": "culture-research/1.0"})

    for month in months:
        for lang, n_keep in (("en", args.n_en), ("zh", args.n_zh)):
            try:
                files = list_files(client, month, lang)
            except Exception as e:
                print(f"[{month}/{lang}] 列表失败: {e}")
                continue
            pick = files if len(files) <= n_keep else sorted(rng.sample(files, n_keep))
            print(f"[{month}/{lang}] 共 {len(files)} 分片, 下载 {len(pick)} 个…")
            rows_all, got = [], 0
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = {ex.submit(download, client, month, lang, f): f for f in pick}
                for fut in as_completed(futs):
                    f = futs[fut]
                    try:
                        path = fut.result()
                        got += 1
                        df = pd.read_parquet(path)
                        meta = df["metadata"].apply(json.loads if df["metadata"].dtype != object
                                                    else (lambda m: m))
                        # metadata 已是 dict
                        meta = df["metadata"]
                        flat = pd.DataFrame({
                            "url": meta.map(lambda m: m.get("url", "")),
                            "hostname": meta.map(lambda m: m.get("hostname", "")),
                            "sitename": meta.map(lambda m: m.get("sitename", "")),
                            "date": meta.map(lambda m: m.get("date", "")),
                            "title": meta.map(lambda m: m.get("title", "")),
                            "description": meta.map(lambda m: m.get("description", "")),
                            "year_month": meta.map(lambda m: m.get("year_month", "")),
                        })
                        df = pd.concat([df[["text", "id"]].reset_index(drop=True),
                                        flat.reset_index(drop=True)], axis=1)
                        hit = df.apply(lambda r: host_match(r["url"], r["hostname"]), axis=1)
                        df = df[hit.notna()].copy()
                        if len(df):
                            df[["outlet", "culture"]] = pd.DataFrame(
                                df.apply(lambda r: host_match(r["url"], r["hostname"]),
                                         axis=1).tolist(), index=df.index)
                            df["text"] = df["text"].map(lambda t: " ".join(str(t).split()))
                            df = df[df["text"].str.len() >= args.min_len]
                            rows_all.append(df)
                        print(f"  {f}: 累计命中 {sum(len(x) for x in rows_all)}")
                    except Exception as e:
                        print(f"  {f} 失败: {e}")
            if rows_all:
                out = FIL / f"{lang}_{month}.parquet"
                res = pd.concat(rows_all, ignore_index=True)
                res.to_parquet(out, index=False)
                print(f"== {month}/{lang}: 保存 {len(res):,} 篇 -> {out.name}")
                print("   媒体分布:", res["outlet"].value_counts().to_dict())
            time.sleep(0.5)


if __name__ == "__main__":
    main()
