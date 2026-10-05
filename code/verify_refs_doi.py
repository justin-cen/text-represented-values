# -*- coding: utf-8 -*-
"""OpenAlex 权威核验（DOI 优先）：
1) 条目含 DOI → 直接按 DOI 精确查询（权威匹配）；
2) 无 DOI（多为专著）→ 标题检索 + 作者/年份交叉验证；
输出 work/ref_verification2.csv，并标出需人工关注的条目。
"""
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(r"E:\culture-difference")
KEYS = (BASE / ".secrets" / "openalex-keys.txt").read_text(encoding="utf-8").split()
OUT = BASE / "work" / "ref_verification2.csv"
PAPERS = {
    "main": BASE / "work" / "paper" / "source.xmd",
    "A": BASE / "work" / "paper" / "paperA_三层共识.xmd",
    "B": BASE / "work" / "paper" / "paperB_english.xmd",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\)\]]+")


def parse_refs(path):
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"@ \[(\d+)\]\s*(.+)", line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def title_of(raw):
    """APA 式：作者段 (年). 标题. 期刊…  → 取 (年) 之后的第一段（标题可能含句点则多取一段）。"""
    s = re.sub(r"\*", "", raw)
    s = re.sub(r"https?://\S+", "", s)
    m = re.search(r"[\(（](\d{4})[\)）]\.?\s*", s)
    year = int(m.group(1)) if m else None
    tail = s[m.end():] if m else s
    parts = [p.strip() for p in re.split(r"\.\s+", tail) if p.strip()]
    if not parts:
        return tail.strip(" ."), year
    t = parts[0]
    # 若首段过短（如 "Can we fix it?"），并入下一段
    if len(t) < 25 and len(parts) > 1:
        t = t + ". " + parts[1]
    return t.strip(" ."), year


def first_author(raw):
    m = re.match(r"\s*([A-Z][A-Za-zÀ-ÿ'’\-]+)", raw)
    return m.group(1) if m else ""


def api(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dsh-agent (research)"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"__notfound__": True}
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(2 * (i + 1)); last = f"HTTP {e.code}"; continue
            return {"__error__": f"HTTP {e.code}"}
        except Exception as e:
            last = type(e).__name__
            if i < tries - 1:
                time.sleep(2 * (i + 1)); continue
            return {"__error__": last}
    return {"__error__": last}


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (s or "").lower())


def fmt(w):
    src = (w.get("primary_location") or {}).get("source") or {}
    return {
        "title_oa": w.get("title"),
        "year_oa": w.get("publication_year"),
        "venue": src.get("display_name"),
        "publisher": src.get("host_organization_name"),
        "type": w.get("type"),
        "cited_by": w.get("cited_by_count"),
        "doi_oa": (w.get("doi") or "").replace("https://doi.org/", ""),
        "authors_oa": "; ".join(a["author"]["display_name"]
                                for a in (w.get("authorships") or [])[:3]),
    }


def main():
    rows, ki = [], 0
    for pname, path in PAPERS.items():
        refs = parse_refs(path)
        print(f"\n=== {pname} ({len(refs)} 条) ===")
        for n, raw in refs:
            key = KEYS[ki % len(KEYS)]
            doi = DOI_RE.search(raw)
            rec = {"论文": pname, "编号": n}
            if doi:
                d = api(f"https://api.openalex.org/works/https://doi.org/{doi.group(0)}?api_key={key}")
                if d.get("__notfound__"):
                    rec.update({"方式": "DOI", "状态": "DOI 未收录", "doi_ref": doi.group(0)})
                elif "__error__" in d:
                    rec.update({"方式": "DOI", "状态": d["__error__"], "doi_ref": doi.group(0)})
                else:
                    rec.update({"方式": "DOI", "状态": "精确匹配", "doi_ref": doi.group(0), **fmt(d)})
            else:
                t, y = title_of(raw)
                q = urllib.parse.quote(t[:170])
                d = api(f"https://api.openalex.org/works?search={q}&per_page=5&api_key={key}")
                res = (d.get("results") or []) if "__error__" not in d else []
                if not res:
                    rec.update({"方式": "标题", "状态": d.get("__error__", "未找到"), "推定标题": t})
                else:
                    tn, fa = norm(t), norm(first_author(raw))
                    best, bs = None, -9
                    for w in res:
                        wn = norm(w.get("title"))
                        s = 0.0
                        if tn and wn and (tn in wn or wn in tn):
                            s += 2.0
                        if y and w.get("publication_year") and abs(w["publication_year"] - y) <= 1:
                            s += 1.0
                        au = norm(" ".join(a["author"]["display_name"]
                                           for a in (w.get("authorships") or [])[:3]))
                        if fa and fa in au:
                            s += 0.5
                        if s > bs:
                            best, bs = w, s
                    rec.update({"方式": "标题", "状态": "标题匹配" if bs >= 2.5 else "需人工核对",
                                "推定标题": t, **fmt(best)})
            rec["原条目"] = raw[:150]
            rows.append(rec)
            print(f"  [{n:>2}] {rec.get('状态',''):<10} {str(rec.get('year_oa') or '-'):<6} "
                  f"{(rec.get('venue') or '-')[:36]:<36} 被引 {rec.get('cited_by') if rec.get('cited_by') is not None else '-'}")
            time.sleep(0.1)
            # 任何异常状态都轮换密钥（401/429/未收录等）
            if rec.get("方式") == "DOI":
                if rec.get("状态") != "精确匹配":
                    ki += 1
            elif rec.get("状态") != "标题匹配":
                ki += 1
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\n=== 汇总 ===")
    print(df["状态"].value_counts().to_string())
    print(f"\n总计 {len(df)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
