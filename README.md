# Text-Represented Values: Measuring Cultural Values across State, Media, and Public Discourse

**文本表征的价值观：国家、媒体与大众三层话语中的文化差异测量**

This repository contains the reproducible code, frozen bilingual lexicon, aggregate results, figures, and manuscripts for a research programme that measures cultural values in large-scale Chinese and Western text using a **frozen bilingual lexicon + multilingual embedding** pipeline.

本仓库收录"冻结双语词库 + 多语嵌入"文本价值观测量研究 programme 的可复现代码、冻结词库、聚合结果、图与论文手稿。

---

## Contents / 目录

| Path | Description |
| --- | --- |
| [`code/`](code) | Full reproducible pipeline (52 scripts): corpus construction → paired-set matching → lexicon freezing → embedding → analysis → robustness checks → figures. See [`code/代码说明.md`](code/代码说明.md). |
| [`lexicon/`](lexicon) | Frozen bilingual lexicon v3 (2,559 words, 4 value systems / 56 facets), official anchors (VSM2013, WVS-7, SVS/PVQ-RR), Socialist Core Values definitions, and the 51-word adjudication record. |
| [`results/`](results) | Aggregate result tables (contrast scores, robustness checks, criterion correlations, layer-consensus statistics). **No raw text.** |
| [`figures/`](figures) | All paper figures (PNG, CJK rendered with SimSun + Times New Roman). |
| [`papers/`](papers) | Three manuscripts (`.docx` / `.pdf` / `.xmd` source). |
| [`data_说明.md`](data_说明.md) | Data dictionary and processing notes. |

## The three manuscripts / 三篇论文

1. **Main paper (基础论文)** — *文本表征的价值观——国家、媒体与大众三层话语中的中西文化差异测量研究*. A measurement-validity study: a four-fold robustness framework (lexicon-version sensitivity, within-culture placebo, three-model triangulation, questionnaire-item anchoring) applied to three content-matched corpora.
2. **Paper A (国内)** — *国家、媒体与大众的三层共识：基于大规模文本测量的中国价值观结构与十年稳定性*. Value-consensus quantification across the state, media, and public layers.
3. **Paper B (international)** — *Measuring Cultural Values in Text without a Cultural Baseline*. Methodological contribution: cross-lingual baseline-bias correction, the four-fold robustness framework, and an explicit construct boundary (text measures discourse register, not national attitudes).

## Method in brief / 方法概要

Five steps: (1) vectorize texts and lexicon words with one multilingual encoder; (2) locate facet centroids as the mean of constituent word vectors; (3) score texts by cosine proximity to each facet; (4) normalize **within text** to remove the cross-lingual baseline bias; (5) reduce dimensions to pole-contrast scores.

$$\text{contrast}(t,D) = s(t,D^{+}) - s(t,D^{-})$$

The key methodological finding is that raw cross-lingual cosine similarity is **not** comparable across languages: Chinese text scores systematically higher (+0.03–0.06) on *all* 56 facets, including mutually exclusive poles. Within-text normalization removes this component.

## Data availability / 数据可得性

Raw corpora are **not redistributed** here, because the constituent sources are third-party user-generated content subject to platform terms of service (Douban, Letterboxd, IMDb) or are large public archives (FineNews, THUCNews, XSum, aclImdb, Tag Genome). Instead:

- Every corpus is **reconstructible** from the provided scripts plus the documented public sources (see [`data_说明.md`](data_说明.md) and [`code/代码说明.md`](code/代码说明.md)).
- All **derived/aggregate** results needed to verify the papers' numbers are included under [`results/`](results).
- The **frozen lexicon** — our own research artifact — is fully released under [`lexicon/`](lexicon).
- The paired analysis sets (with text) are available from the authors on reasonable request, subject to the source platforms' terms.

## Reproducibility / 复现

```bash
# environment: Python 3.13, pandas / pyarrow / sentence-transformers / trafilatura /
#              matplotlib / scipy / ahocorasick / opencc-python-reimplemented
# models: intfloat/multilingual-e5-large, sentence-transformers/LaBSE, BAAI/bge-m3 (HuggingFace)

python code/embed_facet_similarity.py --set reviews_samefilms_v3 --lexicon v3 --device cpu
python code/analyze_sims.py --set reviews_samefilms_v3
python code/robustness_pack.py          # bootstrap CI, FDR, effect sizes, placebo
python code/cross_model_3way_fast.py    # three-encoder triangulation
python code/item_anchor_check.py        # questionnaire-item anchoring
```

Manuscripts are rendered from Markdown-like `.xmd` sources by a self-contained builder (`docx_builder.py` + `omml_latex.py`), producing Word documents with native OMML equations, three-line tables, and hanging-indent footnotes.

## Robustness framework / 稳健性框架

Every sign-based claim must survive four independent checks plus two data-processing checks:

| Check | What varies |
| --- | --- |
| Lexicon-version sensitivity | three lexicon specifications (anchors / +unscreened expansion / frozen) |
| Eleven-layer replication | 2005–2026, three genres, non-overlapping platform corpora |
| Within-culture placebo | Chinese-vs-Chinese and Western-vs-Western outlet pairs (noise floor) |
| Cross-model triangulation | three architecturally independent encoders (e5, LaBSE, BGE-M3) |
| Questionnaire-item anchoring | official VSM2013 item statements as the anchor |
| Data processing | truncation (full-text chunk-mean vs 2,000-char), same-year alignment, version disambiguation |

## Citation / 引用

If you use the lexicon, code, or results, please cite the corresponding manuscript(s). BibTeX entries will be added upon publication.

## License / 许可

Code: MIT. Lexicon and derived results: CC BY-NC 4.0 (non-commercial research use). Third-party corpora remain under their original terms.
