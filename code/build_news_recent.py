# -*- coding: utf-8 -*-
"""v3: 整合全部近期新闻为统一语料并生成"严格同期"与"近期窗口"两个分析集。

输入:
  raw/finenews_filtered/en_2025-*.parquet   西方媒体 (FineNews, 2025 全年分片抽样)
  raw/scrape_2026/rmrb.parquet              人民日报 (2025-08 ~ 2026-09, 日期URL校正)
  raw/scrape_2026/xinhua.parquet            新华网 (2026-09 前后, 日期URL校正)
  raw/scrape_2026/chinanews.parquet         中新网 (2026 近月, 日期URL校正)
  raw/scrape_2026/cnr.parquet               央广网 (2026 近月, 日期URL校正)
  raw/scrape_2026/chinadaily.parquet        中国日报英文 (2025-08 ~ 2026-09)

输出:
  news_zh_recent/news_zh_recent.parquet     中文近期新闻全集
  news_en_recent/news_en_recent.parquet     英文近期新闻全集 (west + zh_en桥接)
  analysis_sets/news_aligned_2025.parquet   ★严格同期同月配对 (2025-08~12, 月份×主题分层)
  analysis_sets/news_recent_2026.parquet    近期窗口配对 (zh 2026-08~09 vs west 2025-08~10)
"""
import argparse
import hashlib
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference\data_corpus")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TOPIC_EN = {
    "时政": r"\b(government|minister|parliament|election|president|policy|senate|congress|diplomat|referendum|prime minister|white house|downing street)\b",
    "社会": r"\b(police|court|crime|school|hospital|nhs|housing|community|charity|immigration|welfare|protest|strike)\b",
    "财经": r"\b(economy|bank|market|stock|shares|business|trade|inflation|unemployment|growth|budget|tax|investment|pound|euro|dollar)\b",
    "科技": r"\b(technology|internet|digital|software|science|research|computer|smartphone|\bdata\b|ai\b|robot|space|nasa|app)\b",
    "教育": r"\b(university|student|school|education|teacher|exam|college|pupil|academy|degree)\b",
    "娱乐": r"\b(film|movie|music|festival|actor|actress|album|show|television|tv\b|award|singer|concert|star|sport|football|match)\b",
}
TOPIC_ZH = {
    "时政": r"政府|国务院|政策|外交|主席|总书记|总理|人大|政协|党中央|选举|峰会|部长",
    "社会": r"民生|社区|医院|养老|就业|住房|公安|法院|案件|志愿|乡村振兴|社保|健康",
    "财经": r"经济|金融|市场|企业|贸易|投资|消费|股市|银行|产业|GDP|外贸|营商",
    "科技": r"科技|技术|人工智能|数字|网络|航天|科研|创新|芯片|互联网|卫星|5G|6G",
    "教育": r"教育|学生|学校|教师|高考|大学|课程|校园|义务教育|双减",
    "娱乐": r"文化|电影|音乐|演出|艺术|旅游|体育|赛事|明星|电视|票房|非遗|文博",
}


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def bucket(texts: pd.Series, pats: dict) -> pd.Series:
    comp = {k: re.compile(v, re.I) for k, v in pats.items()}
    out = []
    for t in texts:
        hit = "其他"
        for k, p in comp.items():
            if p.search(t):
                hit = k
                break
        out.append(hit)
    return pd.Series(out, index=texts.index)


def uni(df, culture, source, prefix):
    return pd.DataFrame({
        "corpus_id": [f"{prefix}_" + md5(u) for u in df["url"]],
        "culture": culture, "genre": "news", "source": source,
        "channel": df["outlet"], "item": df["url"],
        "date": df["date"], "year": df["date"].str[:4],
        "rating": None, "sentiment": None,
        "title": df.get("title"), "text": df["text"], "meta": None,
    })


def stratified_sample(df: pd.DataFrame, key: str, n: int, seed: int) -> pd.DataFrame:
    keys = df[key].astype(str)
    vc = keys.value_counts()
    alloc = (vc / vc.sum() * n).round().astype(int).clip(lower=1)
    alloc[alloc.idxmax()] += n - alloc.sum()
    parts = []
    for cat, k in alloc.items():
        pool = df[keys == cat]
        parts.append(pool.sample(min(k, len(pool)), random_state=seed))
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-month", type=int, default=1500,
                    help="严格同期集中每月每侧样本量")
    ap.add_argument("--n-2026", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- 载入 ----------
    fn = pd.concat([pd.read_parquet(p)
                    for p in sorted((BASE / "raw" / "finenews_filtered").glob("en_2025-*.parquet"))],
                   ignore_index=True)
    fn = fn.drop_duplicates(subset=["url"]).drop_duplicates(subset=["text"])
    fn = fn[(fn["date"] >= "2025-01-01") & (fn["date"] <= "2025-12-31")]
    print(f"FineNews: {len(fn):,} 篇  {fn['outlet'].value_counts().to_dict()}")

    def url_date_rmrb(u):
        m = re.search(r"/content/(202\d)(\d{2})/(\d{2})/", u)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

    def url_date_xh(u):
        m = re.search(r"/(202\d)(\d{2})(\d{2})/", u)
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

    sc = BASE / "raw" / "scrape_2026"
    rmrb = pd.read_parquet(sc / "rmrb.parquet")
    rmrb["date"] = rmrb["url"].map(url_date_rmrb)
    rmrb["outlet"] = "人民日报"
    xh = pd.read_parquet(sc / "xinhua.parquet")
    xh["date"] = xh["url"].map(url_date_xh)
    xh["outlet"] = "新华网"
    extra = []
    for f, outlet in [("chinanews.parquet", "中新网"), ("cnr.parquet", "央广网")]:
        p = sc / f
        if p.exists():
            d = pd.read_parquet(p)
            d["outlet"] = outlet
            extra.append(d)
    cd = pd.read_parquet(sc / "chinadaily.parquet")

    # ---------- 统一语料 ----------
    west = uni(fn[fn["culture"] == "west"], "west", "finenews", "fn")
    zhen_fn = uni(fn[fn["culture"] != "west"], "zh_en", "finenews", "fn")
    zh_parts = [uni(rmrb.dropna(subset=["date"]), "zh", "rmrb", "rr"),
                uni(xh.dropna(subset=["date"]), "zh", "xinhua", "xh")]
    for d in extra:
        src = "chinanews" if (d["outlet"] == "中新网").all() else "cnr"
        zh_parts.append(uni(d.dropna(subset=["date"]), "zh", src,
                            "cn" if src == "chinanews" else "cr"))
    cd2 = uni(cd, "zh_en", "chinadaily", "cd")

    news_zh = pd.concat(zh_parts, ignore_index=True).drop_duplicates(subset=["text"])
    news_en = pd.concat([west, zhen_fn, cd2], ignore_index=True).drop_duplicates(subset=["text"])
    (BASE / "news_zh_recent").mkdir(exist_ok=True)
    (BASE / "news_en_recent").mkdir(exist_ok=True)
    news_zh.to_parquet(BASE / "news_zh_recent" / "news_zh_recent.parquet", index=False)
    news_en.to_parquet(BASE / "news_en_recent" / "news_en_recent.parquet", index=False)
    print(f"news_zh_recent: {len(news_zh):,}  {news_zh['channel'].value_counts().to_dict()}")
    print(f"  日期: {news_zh['date'].min()} → {news_zh['date'].max()}")
    print(f"news_en_recent: {len(news_en):,}")
    print(f"  日期: {news_en['date'].min()} → {news_en['date'].max()}")

    # ---------- 分析集 A: 月份对齐（人民日报2026-MM vs FineNews西方2025-MM） ----------
    # 注：人民日报电子版仅可回溯至 2026-01，无法覆盖 2025；
    # 故采用相邻年份同月份对齐（季节/议题周期可比），MM=01..09。
    za = news_zh[news_zh["date"] >= "2026-01-01"].copy()
    wa = west[west["date"] >= "2025-01-01"].copy()
    za["month"] = za["date"].str[5:7]
    wa["month"] = wa["date"].str[5:7]
    for df, pats in ((za, TOPIC_ZH), (wa, TOPIC_EN)):
        df["topic"] = bucket(df["text"], pats)
    za, wa = za[za["topic"] != "其他"], wa[wa["topic"] != "其他"]
    parts = []
    for mm in sorted(set(za["month"]) & set(wa["month"])):
        z_m, w_m = za[za["month"] == mm], wa[wa["month"] == mm]
        n = min(args.per_month, len(z_m), len(w_m))
        z_s = stratified_sample(z_m, "topic", n, args.seed)
        w_s = stratified_sample(w_m, "topic", n, args.seed)
        z_s = z_s.assign(match_month=mm)
        w_s = w_s.assign(match_month=mm)
        parts += [z_s, w_s]
        print(f"  月 {mm}: zh(2026) {len(z_m):,}→{len(z_s)}  west(2025) {len(w_m):,}→{len(w_s)}")
    aligned = pd.concat(parts).drop(columns=["topic", "month"])
    out_a = BASE / "analysis_sets" / "news_aligned_month.parquet"
    aligned.to_parquet(out_a, index=False)
    print(f"news_aligned_month: {len(aligned):,} -> {out_a}")

    # ---------- 分析集 B: 近期窗口 2026 ----------
    zb = news_zh[news_zh["date"] >= "2026-08-01"].copy()
    wb = west[(west["date"] >= "2025-08-01") & (west["date"] <= "2025-10-31")].copy()
    zb["topic"] = bucket(zb["text"], TOPIC_ZH)
    wb["topic"] = bucket(wb["text"], TOPIC_EN)
    zb, wb = zb[zb["topic"] != "其他"], wb[wb["topic"] != "其他"]
    n = min(args.n_2026, len(zb), len(wb))
    z_s = stratified_sample(zb, "topic", n, args.seed)
    zh_prop = z_s["topic"].value_counts(normalize=True)
    w_parts = []
    for cat, prop in zh_prop.items():
        k = int(round(prop * n))
        pool = wb[wb["topic"] == cat]
        if len(pool) == 0:
            pool = wb
        w_parts.append(pool.sample(min(k, len(pool)), random_state=args.seed))
    recent26 = pd.concat([z_s.drop(columns=["topic"]),
                          pd.concat(w_parts).drop(columns=["topic"])])
    out_b = BASE / "analysis_sets" / "news_recent_2026.parquet"
    recent26.to_parquet(out_b, index=False)
    print(f"news_recent_2026: {len(recent26):,} -> {out_b}  "
          f"(zh {len(z_s):,} 2026-08~09 / west {len(pd.concat(w_parts)):,} 2025-08~10)")


if __name__ == "__main__":
    main()
