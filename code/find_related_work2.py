# -*- coding: utf-8 -*-
"""OpenAlex 精准主题检索（title_and_abstract.search，短语限定）：
为三篇论文寻找 2024—2026 相关文献，避免通用词召回噪声。
"""
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
KEYS = (Path(r"E:\culture-difference\.secrets\openalex-keys.txt").read_text(encoding="utf-8").split())
OUT = Path(r"E:\culture-difference\work\related_work_candidates.csv")

# 精确短语检索（title_and_abstract.search），归属论文
QUERIES = [
    ('"cultural values" AND "large language models"', "B"),
    ('"cultural alignment"', "B"),
    ('"cross-cultural" AND "large language models"', "B"),
    ('"cultural dimensions" AND "text"', "both"),
    ('"value dimensions" AND "word embeddings"', "B"),
    ('"World Values Survey" AND "text"', "B"),
    ('"socialist core values"', "A"),
    ('"Chinese values" AND "discourse"', "A"),
    ('"government work report"', "A"),
    ('"measurement validity" AND "text"', "both"),
    ('"text as data" AND "culture"', "both"),
    ('"moral values" AND "language models"', "B"),
    ('"cross-lingual" AND "cultural bias"', "B"),
    ('"national culture" AND "text analysis"', "both"),
    ('"value salience" AND "text"', "both"),
    ('"public opinion" AND "embeddings" AND "measurement"', "both"),
]


def api(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsh-agent (research)"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(2 * (i + 1)); continue
            return {"__err__": f"HTTP {e.code}"}
        except Exception as e:
            if i < tries - 1:
                time.sleep(2 * (i + 1)); continue
            return {"__err__": type(e).__name__}
    return {"__err__": "retries"}


rows, seen = [], set()
for i, (q, belongs) in enumerate(QUERIES):
    k = KEYS[i % len(KEYS)]
    url = ("https://api.openalex.org/works"
           f"?filter=title_and_abstract.search:{urllib.parse.quote(q)},"
           "from_publication_date:2023-01-01"
           f"&sort=cited_by_count:desc&per_page=10&api_key={k}")
    d = api(url)
    res = d.get("results") or []
    print(f"\n=== {q[:56]}  [{belongs}] → {len(res)} ===")
    for w in res:
        wid = w.get("id")
        if wid in seen:
            continue
        seen.add(wid)
        src = (w.get("primary_location") or {}).get("source") or {}
        rows.append({
            "检索式": q, "归属": belongs, "标题": w.get("title"),
            "年份": w.get("publication_year"), "期刊/会议": src.get("display_name"),
            "类型": w.get("type"), "被引": w.get("cited_by_count"),
            "作者": "; ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])[:2]),
            "DOI": (w.get("doi") or "").replace("https://doi.org/", ""),
        })
        print(f"  [{w.get('cited_by_count'):>4}] {w.get('publication_year')} "
              f"{(src.get('display_name') or '-')[:32]:<32} {str(w.get('title'))[:64]}")
    time.sleep(0.2)

df = pd.DataFrame(rows).sort_values("被引", ascending=False)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n候选（去重）: {len(df)} -> {OUT}")
