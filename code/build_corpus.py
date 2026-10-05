# -*- coding: utf-8 -*-
"""将 data_corpus/raw 下的原始数据整理为统一格式的分析语料。

统一 schema (parquet):
  corpus_id : 唯一ID（源前缀+序号）
  culture   : zh / west
  genre     : review / news
  source    : douban / imdb / thucnews / xsum
  channel   : 频道或栏目（THUCNews类别 / BBC / 电影名）
  item      : 评论对象（电影名）或新闻ID
  date      : 日期（豆瓣评论有，其余为 None）
  year      : 年份（可推导时）
  rating    : 数值评分（豆瓣1-5星，其余 None）
  sentiment : pos / neg / None（IMDB标注，豆瓣可由rating推导但不预设）
  title     : 新闻标题（影评为 None）
  text      : 正文文本
  meta      : 其余元数据 JSON 字符串

用法: python build_corpus.py [--min-review-len 50] [--min-news-len 200]
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
RAW = BASE / "raw"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def norm_text(s) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split())


# ---------------------------------------------------------------- douban
def build_douban(min_len: int) -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "douban").glob("train-*.parquet")):
        df = pd.read_parquet(p, columns=["Movie_Name", "Username", "Date", "Star",
                                         "Comment", "Like", "Craw_Date"])
        df["text"] = df["Comment"].map(norm_text)
        df = df[df["text"].str.len() >= min_len]
        frames.append(df)
        print(f"  {p.name}: 原始 {len(pd.read_parquet(p, columns=['ID'])):,} 行, "
              f"过滤后 {len(df):,} 行")
    df = pd.concat(frames, ignore_index=True)
    df["Username"] = df["Username"].fillna("").astype(str)
    df["Movie_Name"] = df["Movie_Name"].fillna("").astype(str)
    df["Star"] = pd.to_numeric(df["Star"], errors="coerce")
    df["Like"] = df["Like"].fillna(0)
    before = len(df)
    df = df.drop_duplicates(subset=["Username", "Movie_Name", "text"])
    df = df[df["Star"].between(1, 5)]
    print(f"  douban 去重 {before:,} -> {len(df):,}")
    out = pd.DataFrame({
        "corpus_id": ["db_" + md5(u + m + t) for u, m, t in
                      zip(df["Username"], df["Movie_Name"], df["text"])],
        "culture": "zh", "genre": "review", "source": "douban",
        "channel": df["Movie_Name"], "item": df["Movie_Name"],
        "date": df["Date"],
        "year": pd.to_datetime(df["Date"], errors="coerce").dt.year,
        "rating": df["Star"].astype("float64"),
        "sentiment": None, "title": None, "text": df["text"],
        "meta": [json.dumps({"like": l, "craw_date": c}, ensure_ascii=False)
                 for l, c in zip(df["Like"], df["Craw_Date"])],
    })
    return out


# ---------------------------------------------------------------- imdb
def build_imdb(min_len: int) -> pd.DataFrame:
    frames = []
    for split in ["train", "test", "unsupervised"]:
        p = RAW / "imdb" / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(p)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].str.replace(r"<br\s*/?>", " ", regex=True)
    df["text"] = df["text"].map(norm_text)
    df = df[df["text"].str.len() >= min_len]
    df = df.drop_duplicates(subset=["text"])
    label_map = {1: "pos", 0: "neg", -1: None}
    out = pd.DataFrame({
        "corpus_id": ["im_" + md5(t) for t in df["text"]],
        "culture": "west", "genre": "review", "source": "imdb",
        "channel": None, "item": None,
        "date": None, "year": None, "rating": None,
        "sentiment": df["label"].map(label_map),
        "title": None, "text": df["text"],
        "meta": [json.dumps({"split": s}) for s in df["split"]],
    })
    return out


# ---------------------------------------------------------------- thucnews
def build_thucnews(min_len: int) -> pd.DataFrame:
    frames = []
    for p in sorted((RAW / "thucnews").glob("*.jsonl")):
        cat = p.stem
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        df["text"] = df["content"].map(norm_text)
        df = df[df["text"].str.len() >= min_len]
        df["channel"] = cat
        frames.append(df[["id", "title", "text", "channel"]])
        print(f"  THUCNews[{cat}]: {len(df):,} 行 (>= {min_len} 字符)")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    out = pd.DataFrame({
        "corpus_id": ["th_" + md5(t) for t in df["text"]],
        "culture": "zh", "genre": "news", "source": "thucnews",
        "channel": df["channel"], "item": df["id"],
        "date": None, "year": None, "rating": None,
        "sentiment": None, "title": df["title"], "text": df["text"],
        "meta": None,
    })
    return out


# ---------------------------------------------------------------- xsum
def build_xsum(min_len: int) -> pd.DataFrame:
    frames = []
    for split in ["train", "validation", "test"]:
        p = RAW / "xsum" / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(p)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["document"].map(norm_text)
    df = df[df["text"].str.len() >= min_len]
    df = df.drop_duplicates(subset=["text"])
    out = pd.DataFrame({
        "corpus_id": ["xs_" + str(i) for i in df["id"]],
        "culture": "west", "genre": "news", "source": "xsum",
        "channel": "BBC", "item": df["id"].astype(str),
        "date": None, "year": None, "rating": None,
        "sentiment": None, "title": df["summary"].map(norm_text),
        "text": df["text"],
        "meta": [json.dumps({"split": s}) for s in df["split"]],
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-review-len", type=int, default=50)
    ap.add_argument("--min-news-len", type=int, default=200)
    args = ap.parse_args()

    builders = [
        ("reviews_zh", build_douban, args.min_review_len),
        ("reviews_en", build_imdb, args.min_review_len),
        ("news_zh", build_thucnews, args.min_news_len),
        ("news_en", build_xsum, args.min_news_len),
    ]
    stats = {}
    for outdir, fn, min_len in builders:
        print(f"=== 构建 {outdir} (min_len={min_len}) ===")
        df = fn(min_len)
        od = BASE / outdir
        od.mkdir(exist_ok=True)
        out = od / f"{outdir}.parquet"
        df.to_parquet(out, index=False)
        stats[outdir] = len(df)
        lens = df["text"].str.len()
        print(f"  -> {out}  rows={len(df):,}  "
              f"文本长度 mean={lens.mean():.0f} p50={lens.median():.0f} "
              f"p90={lens.quantile(0.9):.0f}")
    print("\n汇总:", json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
