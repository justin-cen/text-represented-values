# -*- coding: utf-8 -*-
"""词库冻结管线 v3：锚定词 + LLM扩充 → 机器筛查 → 冻结词库 + 人工复核表。

筛查项（逐词）：
  1. coherence       与所属刻面质心（不含自身）的余弦 —— 语义内聚
  2. discrimination  与本体系最近邻刻面质心之差(margin>0) —— 刻面区分度
  3. alignment       中文词与即刻面英文词场最近邻余弦（跨语言对齐参考分）
  4. freq_*          在四个语料子样本中的出现次数（Aho-Corasick）
规则：auto_keep = coherence≥P25(即刻面内) 且 margin>0 且 (freq_zh_total≥15 或 freq_en_total≥15)
     其余进 review 表并标注原因；全部结果留痕供人工复核改判。

用法: python freeze_lexicon.py [--freq-sample 60000] [--min-freq 15]
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
LEX = BASE / "work" / "lexicon"
EXP = BASE / "work"
OUT = LEX / "frozen_v3"
OUT.mkdir(exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TAU_ALIGN = 0.85   # zh 词与即刻面英文词场最近邻余弦的参考线（仅报告，不强制）

# 繁→简统一（锚定词表混有繁体，简体语料中频度为0）
try:
    from opencc import OpenCC
    _CC = OpenCC("t2s")
    _n_conv = [0]

    def s2(w: str) -> str:
        if not w:
            return w
        out = _CC.convert(w)
        if out != w:
            _n_conv[0] += 1
        return out
except Exception:                                    # pragma: no cover
    def s2(w: str) -> str:
        return w


# ---------------------------------------------------------------- 载入
def load_all():
    """返回 {system: {facet: {'en_anchor':[], 'zh_anchor':[], 'en_exp':[], 'zh_exp':[]}}}"""
    systems = {}
    # Hofstede
    ha = json.load(open(LEX / "hofstede_anchor.json", encoding="utf-8"))
    ht = json.load(open(LEX / "hofstede_v2_translated_verified_e5.json", encoding="utf-8"))
    ex = json.load(open(EXP / "lexicon_expansion_hofstede.json", encoding="utf-8"))
    systems["hofstede"] = {}
    for pole, rec in ha.items():
        if pole == "meta":
            continue
        systems["hofstede"][pole] = {
            "en_anchor": rec.get("anchor_en", []), "zh_anchor": ht["zh"].get(pole, []),
            "en_exp": ex.get(pole, {}).get("en", []), "zh_exp": ex.get(pole, {}).get("zh", [])}
    # WVS
    we = json.load(open(LEX / "wvs_anchor_en.json", encoding="utf-8"))
    wt = json.load(open(LEX / "wvs_anchor_translated_raw.json", encoding="utf-8"))
    ex = json.load(open(EXP / "lexicon_expansion_wvs.json", encoding="utf-8"))
    systems["wvs"] = {}
    for f, words in we.items():
        if f == "meta":
            continue
        systems["wvs"][f] = {
            "en_anchor": words, "zh_anchor": wt["zh"].get(f, []),
            "en_exp": ex.get(f, {}).get("en", []), "zh_exp": ex.get(f, {}).get("zh", [])}
    # Schwartz
    se = json.load(open(LEX / "schwartz_anchor_en.json", encoding="utf-8"))
    st = json.load(open(LEX / "schwartz_anchor_translated_raw.json", encoding="utf-8"))
    ex = json.load(open(EXP / "lexicon_expansion_schwartz.json", encoding="utf-8"))
    systems["schwartz"] = {}
    for f, words in se.items():
        if f == "meta":
            continue
        systems["schwartz"][f] = {
            "en_anchor": words, "zh_anchor": st["zh"].get(f, []),
            "en_exp": ex.get(f, {}).get("en", []), "zh_exp": ex.get(f, {}).get("zh", [])}
    # SCV
    scv = json.load(open(LEX / "socialist_core_values.json", encoding="utf-8"))
    ex = json.load(open(EXP / "lexicon_expansion_scv.json", encoding="utf-8"))
    systems["scv"] = {}
    for k, v in scv.items():
        if k == "meta":
            continue
        systems["scv"][k] = {
            "en_anchor": v.get("anchor_en", []) or [v.get("en", "")],
            "zh_anchor": v.get("anchor_zh", []),
            "en_exp": ex.get(k, {}).get("en", []), "zh_exp": ex.get(k, {}).get("zh", [])}
    # 全部中文词统一为简体（锚定词表混有繁体）
    for sysname, facets in systems.items():
        for facet, v in facets.items():
            for key in ("zh_anchor", "zh_exp"):
                v[key] = [w for w in dict.fromkeys(s2(x) for x in v[key]) if w]
    print(f"[预处理] 繁→简转换 {_n_conv[0]} 个中文词条")
    return systems


# ---------------------------------------------------------------- 频度
def count_freq(terms_zh, terms_en, sample):
    """terms: 列表；sample: {corpus: [text,...]}。返回 {term: {corpus: count}}"""
    import ahocorasick
    res = {t: {} for t in terms_zh + terms_en}
    # zh 子串匹配；en 词边界匹配（文本两侧补空格后用 ' word ' 单模式，精确计数）
    A = ahocorasick.Automaton()
    for t in terms_zh:
        A.add_word(t, t)
    for t in terms_en:
        A.add_word(" " + t.lower() + " ", t)
    A.make_automaton()
    for corpus, texts in sample.items():
        for text in texts:
            tl = (" " + text + " ") if corpus.endswith("zh") else \
                 (" " + text.lower() + " ")
            for _, t in A.iter(tl):
                res[t][corpus] = res[t].get(corpus, 0) + 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq-sample", type=int, default=60000)
    ap.add_argument("--min-freq", type=int, default=15)
    ap.add_argument("--device", default="cpu",
                    help="嵌入设备；GPU 散热异常期间一律用 cpu")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU 线程上限（避免占满机器）")
    args = ap.parse_args()
    if args.device == "cpu":
        import torch
        torch.set_num_threads(args.threads)
        print(f"CPU 线程数限制为 {args.threads}")

    systems = load_all()
    n0 = sum(len(set(v["en_anchor"] + v["zh_anchor"])) for s in systems.values() for v in s.values())
    n1 = sum(len(set(v["en_exp"] + v["zh_exp"])) for s in systems.values() for v in s.values())
    print(f"锚定词合计 ~{n0}，扩充候选合计 ~{n1}")

    # ---------- 去重（刻面内 + 对锚定词） ----------
    rows = []
    for sysname, facets in systems.items():
        for facet, v in facets.items():
            seen = set(x.lower() for x in v["en_anchor"]) | set(v["zh_anchor"])
            for lang in ("en", "zh"):
                for w in v[f"{lang}_anchor"]:
                    rows.append(dict(system=sysname, facet=facet, lang=lang,
                                     word=w, source="anchor"))
            for lang in ("en", "zh"):
                for w in v[f"{lang}_exp"]:
                    key = w.lower() if lang == "en" else w
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(dict(system=sysname, facet=facet, lang=lang,
                                     word=w, source="llm"))
    cand = pd.DataFrame(rows)
    print(f"进入筛查的候选: {len(cand):,}（含锚定词 {(cand.source=='anchor').sum():,}）")

    # ---------- 刻面内跨刻面去重：同一 system 内同词多刻面 → 留给语义最近刻面 ----------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=args.device)
    print(f"嵌入设备: {args.device}")

    def emb(texts):
        return model.encode(["query: " + t for t in texts], normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=512, show_progress_bar=False)

    # 先嵌全部词（带磁盘缓存，避免重复计算）
    word_cache = OUT / f"word_emb_{args.device}.npz"
    words = cand["word"].tolist()
    if word_cache.exists():
        z = np.load(word_cache, allow_pickle=True)
        if list(z["words"]) == words:
            E = z["emb"].astype(np.float32)
            print(f"词向量缓存命中: {E.shape}")
        else:
            E = emb(words)
            np.savez_compressed(word_cache, words=np.array(words, dtype=object), emb=E.astype(np.float16))
    else:
        E = emb(words)
        np.savez_compressed(word_cache, words=np.array(words, dtype=object), emb=E.astype(np.float16))
    cand["vec"] = list(E)
    # 刻面质心（用锚定词构建）
    cent = {}
    for sysname, facets in systems.items():
        for facet, v in facets.items():
            ws = [x for x in v["en_anchor"] + v["zh_anchor"] if x]
            if ws:
                c = emb(ws).mean(axis=0)
                cent[(sysname, facet)] = c / np.linalg.norm(c)
    sysfacets = {}
    for (s, f), c in cent.items():
        sysfacets.setdefault(s, []).append(f)

    # 英文词场向量：按 (system, facet) 预计算一次（原先在循环内重复编码，CPU 上极慢）
    en_field = {}
    for sysname, facets in systems.items():
        for facet, v in facets.items():
            ws = [x for x in v["en_anchor"] + v["en_exp"] if x]
            if ws:
                ee = emb(ws)
                en_field[(sysname, facet)] = ee
    print(f"英文词场预计算完成: {len(en_field)} 个 (system, facet)")

    dup_groups = cand.groupby(["system", cand["word"].str.lower()])
    drop_idx = []
    for (sysname, _), g in dup_groups:
        if len(g) <= 1:
            continue
        # 同词多刻面：保留与质心最相似者
        sims = []
        for i, r in g.iterrows():
            c = cent.get((sysname, r["facet"]))
            sims.append(float(np.dot(r["vec"], c)) if c is not None else -1)
        keep = int(np.argmax(sims))
        drop_idx += [g.index[j] for j in range(len(g)) if j != keep]
    cand = cand.drop(index=drop_idx)
    print(f"跨刻面重复剔除: {len(drop_idx)}，剩余 {len(cand):,}")

    # ---------- coherence / discrimination / alignment ----------
    coh, marg, align = [], [], []
    for i, r in cand.iterrows():
        own = cent[(r["system"], r["facet"])]
        v = r["vec"]
        s_own = float(np.dot(v, own))
        coh.append(s_own)
        others = [float(np.dot(v, cent[(r["system"], f)]))
                  for f in sysfacets[r["system"]] if f != r["facet"]]
        marg.append(s_own - (max(others) if others else -1))
        if r["lang"] == "zh":
            ee = en_field.get((r["system"], r["facet"]))
            align.append(float((ee @ v).max()) if ee is not None else np.nan)
        else:
            align.append(np.nan)
    cand["coherence"], cand["margin"], cand["align_en"] = coh, marg, align

    # coherence 阈值：即刻面内 P25（锚定词+候选混合分布）
    def p25(s):
        return np.percentile(s, 25)
    thr = cand.groupby(["system", "facet"])["coherence"].transform(p25)
    cand["pass_coh"] = cand["coherence"] >= thr - 1e-9
    cand["pass_margin"] = cand["margin"] > 0

    # ---------- 语料频度 ----------
    print("统计语料频度（抽样）…")
    rng = np.random.RandomState(42)
    sample = {}
    for name, path in [("reviews_zh", "reviews_zh/reviews_zh.parquet"),
                       ("reviews_en", "reviews_en/reviews_en_letterboxd.parquet"),
                       ("news_zh", "news_zh_recent/news_zh_recent.parquet"),
                       ("news_en", "news_en_recent/news_en_recent.parquet")]:
        df = pd.read_parquet(BASE / "data_corpus" / path, columns=["text"])
        n = min(args.freq_sample, len(df))
        sample[name] = df["text"].sample(n, random_state=42).tolist()
        print(f"  {name}: {n:,} 篇")
    terms_zh = cand.loc[cand.lang == "zh", "word"].tolist()
    terms_en = cand.loc[cand.lang == "en", "word"].tolist()
    freq = count_freq(terms_zh, terms_en, sample)
    for c in sample:
        cand["freq_" + c] = cand["word"].map(lambda w: freq.get(w, {}).get(c, 0))
    cand["freq_total"] = cand[[f"freq_{c}" for c in sample]].sum(axis=1)
    cand["pass_freq"] = cand["freq_total"] >= args.min_freq

    # ---------- 自动判定 ----------
    def decide(r):
        if r["source"] == "anchor":
            return "keep"          # 锚定词默认保留（已被筛查标记供复核）
        ok = r["pass_coh"] and r["pass_margin"] and r["pass_freq"]
        return "keep" if ok else "drop"
    cand["auto"] = cand.apply(decide, axis=1)

    def reason(r):
        rs = []
        if not r["pass_coh"]:
            rs.append("内聚低")
        if not r["pass_margin"]:
            rs.append("区分度低")
        if not r["pass_freq"]:
            rs.append("频度低")
        return ";".join(rs) if rs else "通过"
    cand["reason"] = cand.apply(reason, axis=1)

    # ---------- 冻结与复核表 ----------
    frozen = {}
    for (sysname, facet), g in cand[cand["auto"] == "keep"].groupby(["system", "facet"]):
        frozen.setdefault(sysname, {})[facet] = {
            "en": sorted(g.loc[g.lang == "en", "word"].str.lower().unique().tolist()),
            "zh": sorted(g.loc[g.lang == "zh", "word"].unique().tolist())}
    meta = {"version": "v3-frozen-auto", "built": "2026-09-28",
            "rule": f"anchor默认保留；LLM候选需 coherence≥P25 & margin>0 & freq≥{args.min_freq}",
            "note": "auto 判定仅供机器预筛，最终以人工复核 lexicon_v3_review.csv 为准"}
    json.dump({"meta": meta, **frozen}, open(OUT / "lexicon_v3_frozen.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    review = cand.drop(columns=["vec"]).sort_values(
        ["system", "facet", "lang", "auto", "freq_total"],
        ascending=[True, True, True, False, False])
    review.to_csv(OUT / "lexicon_v3_review.csv", index=False, encoding="utf-8-sig")

    print("\n=== 冻结结果（keep/total）===")
    for sysname, facets in systems.items():
        tot = len(cand[cand.system == sysname])
        kp = len(cand[(cand.system == sysname) & (cand.auto == "keep")])
        print(f"  {sysname}: {kp}/{tot}")
    print(f"-> {OUT / 'lexicon_v3_frozen.json'}")
    print(f"-> {OUT / 'lexicon_v3_review.csv'}  ({len(review):,} 行)")
    print("\n频度最低的保留词（检查）:")
    kept = cand[cand.auto == "keep"].nsmallest(8, "freq_total")
    print(kept[["system", "facet", "lang", "word", "freq_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
