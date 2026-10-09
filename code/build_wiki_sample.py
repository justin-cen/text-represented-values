# -*- coding: utf-8 -*-
"""P3 泛化检验：跨语言基线偏置是否普遍存在（22 语种 Wikipedia 样本）。

每语种抽样 N 篇 → 构造成 analysis_set → 用同一冻结词库嵌入 → 计算各语种对各刻面的
平均相似度（绝对，未做文本内标准化），检验：
(a) 是否存在语种层面的系统性偏移（对所有刻面同向）；
(b) 偏移幅度是否因语种而异（→ 跨语言比较必须校正）。
"""
import io
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"E:\culture-difference")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
N = 150

rows = []
W = BASE / "work/data/wiki"
for f in sorted(W.glob("sample_*.parquet")):
    lang = f.stem.replace("sample_", "")
    d = pd.read_parquet(f)
    d = d[d.text.astype(str).str.len() >= 200]
    take = d.sample(n=min(N, len(d)), random_state=42)
    for i, (_, r) in enumerate(take.iterrows()):
        rows.append({"corpus_id": f"wiki_{lang}_{i:04d}", "culture": lang, "genre": "wiki",
                     "source": "wikipedia", "channel": lang, "item": str(r.get("title", ""))[:100],
                     "date": None, "year": 2024, "rating": None, "sentiment": None,
                     "title": str(r.get("title", ""))[:100], "text": str(r["text"]),
                     "meta": str(r.get("country", ""))})

out = pd.DataFrame(rows)
out.to_parquet(BASE / "data_corpus/analysis_sets/wiki_multilang.parquet", index=False)
print(f"-> wiki_multilang.parquet: {len(out):,} 篇 / {out.channel.nunique()} 语种")
print(out.groupby("channel").size().to_string())
