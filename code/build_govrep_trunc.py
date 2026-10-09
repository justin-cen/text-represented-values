# -*- coding: utf-8 -*-
"""政府报告截断版 analysis_set（前 3000 字），用于快速嵌入。"""
import io
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TXT = BASE / "work/data/gov_reports/txt"
rows = []
for f in sorted(TXT.glob("*.txt")):
    m = re.match(r"(.+?)_(\d{4})\.txt$", f.name)
    if not m:
        continue
    prov, year = m.group(1), int(m.group(2))
    t = " ".join(f.read_text(encoding="utf-8", errors="ignore").split())[:3000]
    if len(t) < 500:
        continue
    rows.append({"corpus_id": f"govtrunc_{prov}_{year}", "culture": "zh", "genre": "govreport",
                 "source": "gov_work_report", "channel": prov, "item": f"{prov}{year}年政府工作报告",
                 "date": f"{year}-01-01", "year": year, "rating": None, "sentiment": None,
                 "title": f"{prov}{year}年政府工作报告", "text": t, "meta": None})

d = pd.DataFrame(rows)
d.to_parquet(BASE / "data_corpus/analysis_sets/gov_reports_trunc.parquet", index=False)
print(f"-> gov_reports_trunc.parquet: {len(d)} 篇 / {d.channel.nunique()} 省")
print(f"字符数 中位={int(d.text.str.len().median())}")
