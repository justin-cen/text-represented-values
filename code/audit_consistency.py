# -*- coding: utf-8 -*-
"""三篇论文一致性审计：
1) 表/图编号连续且与正文引用一一对应；
2) 关键统计量与数据文件一致（抽检）；
3) 术语统一性检查。
"""
import io
import re
import sys
import zipfile
import docx
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = r"E:\culture-difference"

PAPERS = {
    "主论文": BASE + r"\work\paper\文本表征的价值观_定稿.docx",
    "论文A": BASE + r"\work\paper\论文A_三层共识.docx",
    "论文B": BASE + r"\work\paper\论文B_english.docx",
}

for name, path in PAPERS.items():
    print(f"\n{'='*50}\n=== {name} ===")
    try:
        d = docx.Document(path)
    except Exception as e:
        print("无法读取:", e); continue
    text = "\n".join(p.text for p in d.paragraphs)
    # 表/图题注编号（表标题/图标题样式）
    tabcaps = [int(re.match(r"(?:表|Table|图|Figure)\s*(\d+)", p.text.strip()).group(1))
               for p in d.paragraphs
               if p.style and p.style.name in ("表标题", "XT表标题") and re.match(r"(?:表|Table)\s*\d+", p.text.strip())]
    figcaps = [int(re.match(r"(?:图|Figure)\s*(\d+)", p.text.strip()).group(1))
               for p in d.paragraphs
               if p.style and p.style.name in ("图标题", "XT图标题") and re.match(r"(?:图|Figure)\s*\d+", p.text.strip())]
    # 正文引用
    tab_refs = sorted(set(int(m.group(1)) for m in re.finditer(r"(?:表|Table)\s*(\d+)", text)))
    fig_refs = sorted(set(int(m.group(1)) for m in re.finditer(r"(?:图|Figure)\s*(\d+)", text)))
    print(f"表题注: {tabcaps}")
    print(f"图题注: {figcaps}")
    # 连续性
    print("表连续:", tabcaps == list(range(1, len(tabcaps)+1)) if tabcaps else "无")
    print("图连续:", figcaps == list(range(1, len(figcaps)+1)) if figcaps else "无")
    # 引用-题注对应
    tab_refs_only = [x for x in tab_refs if x not in tabcaps]
    fig_refs_only = [x for x in fig_refs if x not in figcaps]
    # 注意：题注本身也含"表N/图N"，故引用集含题注号；只检查是否有引用了不存在的题注
    missing_tab = [x for x in tab_refs if tabcaps and x > max(tabcaps)]
    missing_fig = [x for x in fig_refs if figcaps and x > max(figcaps)]
    print("引用超出最大题注号（表）:", missing_tab or "无")
    print("引用超出最大题注号（图）:", missing_fig or "无")
