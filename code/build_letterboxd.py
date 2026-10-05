# -*- coding: utf-8 -*-
"""将 Letterboxd 全量 dump（pkchwy/letterboxd-all-movie-data, 2025-07 抓取）
整理为统一格式的近期英文影评语料 reviews_en_letterboxd.parquet。

过滤规则：评论 >= 200 字符（长评论）、英文（ASCII 字母占比 >= 0.85）、去重。
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
SRC = BASE / "raw" / "letterboxd" / "full_dump.jsonl"
OUT = BASE / "reviews_en" / "reviews_en_letterboxd.parquet"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MIN_LEN = 50


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def is_english(s: str) -> bool:
    if not s:
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters == 0:
        return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in s)
    return ascii_letters / letters >= 0.85


def main():
    rows = []
    n_films = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_films += 1
            if n_films % 100000 == 0:
                print(f"  已扫描 {n_films:,} 部影片, 累积评论 {len(rows):,}")
            try:
                d = json.loads(line)
            except Exception:
                continue
            title = d.get("title") or ""
            year = d.get("year") or ""
            genres = d.get("genres") or []
            furl = d.get("url") or ""
            for rv in d.get("reviews") or []:
                t = " ".join(str(rv.get("review_text") or "").split())
                if len(t) < MIN_LEN or not is_english(t):
                    continue
                rows.append({
                    "user": rv.get("username") or "",
                    "film": title, "film_year": year, "film_url": furl,
                    "genres": ",".join(genres),
                    "likes": rv.get("likes") or "0",
                    "text": t,
                })
    df = pd.DataFrame(rows)
    print(f"扫描完成: {n_films:,} 部影片, 原始长评论 {len(df):,}")
    df = df.drop_duplicates(subset=["user", "film", "text"])
    df = df.drop_duplicates(subset=["text"])
    print(f"去重后: {len(df):,}")
    out = pd.DataFrame({
        "corpus_id": ["lb_" + md5(u + f_ + t) for u, f_, t in
                      zip(df["user"], df["film"], df["text"])],
        "culture": "west", "genre": "review", "source": "letterboxd",
        "channel": df["film"], "item": df["film"] + " (" + df["film_year"] + ")",
        "date": None, "year": None, "rating": None, "sentiment": None,
        "title": None, "text": df["text"],
        "meta": [json.dumps({"user": u, "film_year": y, "film_url": fu,
                             "genres": g, "likes": l}, ensure_ascii=False)
                 for u, y, fu, g, l in
                 zip(df["user"], df["film_year"], df["film_url"], df["genres"], df["likes"])],
    })
    out.to_parquet(OUT, index=False)
    lens = out["text"].str.len()
    print(f"-> {OUT}  rows={len(out):,}  长度 mean={lens.mean():.0f} p50={lens.median():.0f}")


if __name__ == "__main__":
    main()
