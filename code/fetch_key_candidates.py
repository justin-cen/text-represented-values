# -*- coding: utf-8 -*-
"""获取关键候选文献的完整元数据（供补入参考文献表）。"""
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
KEYS = (Path(r"E:\culture-difference\.secrets\openalex-keys.txt").read_text(encoding="utf-8").split())

TARGETS = [
    "Which Humans",
    "On Measurement Validity and Language Models Increasing Validity",
    "Survey of Cultural Awareness in Language Models Text and Beyond",
    "AI Suggestions Homogenize Writing Toward Western Styles and Diminish Cultural Nuance",
    "Whose morality do they speak Unraveling cultural bias in multilingual",
    "Contextualized Construct Representation Leveraging Psychometric",
    "ValuesRAG Enhancing Cultural Alignment Through Retrieval",
    "Investigating Cultural Alignment of Large Language Models",
    "Towards realistic evaluation of cultural value alignment in large language models",
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


for i, t in enumerate(TARGETS):
    k = KEYS[i % len(KEYS)]
    q = urllib.parse.quote(t)
    d = api(f"https://api.openalex.org/works?filter=title.search:{q}&per_page=2&api_key={k}")
    res = d.get("results") or []
    print(f"\n=== {t[:60]} ===")
    if not res:
        d = api(f"https://api.openalex.org/works?search={q}&per_page=2&api_key={k}")
        res = d.get("results") or []
    for w in res[:2]:
        src = (w.get("primary_location") or {}).get("source") or {}
        au = "; ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])[:4])
        print(f"  标题: {w.get('title')}")
        print(f"  年份: {w.get('publication_year')} | 类型: {w.get('type')} | 被引: {w.get('cited_by_count')}")
        print(f"  期刊: {src.get('display_name')} | 出版方: {src.get('host_organization_name')}")
        print(f"  作者: {au}")
        print(f"  DOI : {(w.get('doi') or '').replace('https://doi.org/','')}")
    time.sleep(0.2)
