# -*- coding: utf-8 -*-
"""开干管线：统一 e5 嵌入空间下的价值观刻面相似度计算。

1) 加载四大价值观体系的双语刻面词表（锚定层）：
   - Hofstede 六维 12 极    work/lexicon/hofstede_v2_translated_verified_e5.json (en+zh)
   - WVS 13 刻面             work/lexicon/wvs_anchor_translated_raw.json (en+zh)
   - Schwartz 19 价值观      work/lexicon/schwartz_anchor_translated_raw.json (en+zh)
   - 社会主义核心价值观 12 词  work/lexicon/socialist_core_values.json (zh anchor + en 短语)
2) 每个刻面/极：e5(query:) 嵌入词表 → 双语合并质心（权重=1）；
3) 文本：e5(passage:) 嵌入（默认截取前 2000 字符，后续可加 chunked 均值作稳健性）；
4) 余弦相似度 → 长表 parquet（corpus_id × system × facet × sim）+ 嵌入缓存。

用法:
  python embed_facet_similarity.py --set reviews_samefilms
  python embed_facet_similarity.py --set news_aligned_2025 --batch 96
  python embed_facet_similarity.py --set news_recent_2026 --max-chars 2000
"""
import argparse
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\culture-difference")
CORPUS = BASE / "data_corpus"
LEX = BASE / "work" / "lexicon"
OUT_DIR = CORPUS / "results"
CACHE = CORPUS / "emb_cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ---------------------------------------------------------------- GPU 温控
def gpu_temp() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1.0


def gpu_guard(limit: float, resume: float, abort: float, verbose: bool = True):
    """批间温控：>=limit 暂停散热，>=abort 抛异常中止。"""
    t = gpu_temp()
    if t < 0:
        return
    if t >= abort:
        raise RuntimeError(f"GPU 温度 {t:.0f}°C >= 中止线 {abort:.0f}°C，已安全停止")
    if t >= limit:
        if verbose:
            print(f"  [温控] {t:.0f}°C >= {limit:.0f}°C，暂停散热…", flush=True)
        while t >= resume:
            time.sleep(15)
            t = gpu_temp()
            if t >= abort:
                raise RuntimeError(f"GPU 温度 {t:.0f}°C >= 中止线 {abort:.0f}°C，已安全停止")
        if verbose:
            print(f"  [温控] 降至 {t:.0f}°C，继续", flush=True)


def encode_guarded(model, texts, batch, limit, resume, abort, chunk_save=None,
                   chunk_size=2000, micro=256):
    """微批嵌入 + 微批间主动散热 + 分块落盘（可断点续跑）。

    micro: 每个微批的文本数（默认256，约1-2°C温升量级）；
    微批后若温度 >= limit-2 则主动散热至 resume 以下；>= abort 立即中止。
    """
    n = len(texts)
    n_chunks = (n + chunk_size - 1) // chunk_size
    parts = [None] * n_chunks

    def expected_rows(ci):
        return min(chunk_size, n - ci * chunk_size)

    pending = []
    for i in range(n_chunks):
        p = chunk_save(i) if chunk_save else None
        if p and p.exists():
            a = np.load(p, mmap_mode="r")
            if a.shape[0] == expected_rows(i):
                parts[i] = "cached"
                continue
            print(f"  块 {i} 行数 {a.shape[0]} ≠ 预期 {expected_rows(i)}，判定为残缺，重算",
                  flush=True)
            p.unlink()
        pending.append(i)
    if chunk_save:
        done = n_chunks - len(pending)
        if done:
            print(f"  断点续跑：已完成 {done}/{n_chunks} 块")
    for ci in pending:
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n)
        embs = []
        for m0 in range(lo, hi, micro):
            gpu_guard(limit, resume, abort)
            e = model.encode(texts[m0:min(m0 + micro, hi)], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=batch,
                             show_progress_bar=False)
            embs.append(e)
            t = gpu_temp()
            if t >= abort:
                raise RuntimeError(f"GPU 温度 {t:.0f}°C >= 中止线 {abort:.0f}°C，已安全停止"
                                   "（本块未落盘，续跑时将整片重算）")
            if 0 <= t >= limit - 2:
                print(f"    [散热] {t:.0f}°C，冷却至 {resume:.0f}°C…", flush=True)
                while 0 <= t >= resume:
                    time.sleep(10)
                    t = gpu_temp()
                    if t >= abort:
                        raise RuntimeError(
                            f"GPU 温度 {t:.0f}°C >= 中止线 {abort:.0f}°C，已安全停止")
            else:
                time.sleep(0.3)
        emb = np.vstack(embs)
        if chunk_save:
            np.save(chunk_save(ci), emb.astype(np.float16))
        parts[ci] = emb
        t1 = gpu_temp()
        print(f"  块 {ci + 1}/{n_chunks} ({hi - lo} 文本) 完成"
              + (f"  GPU {t1:.0f}°C" if t1 >= 0 else ""), flush=True)
    if chunk_save:
        parts = [np.load(chunk_save(i)) for i in range(n_chunks)]
    return np.vstack(parts).astype(np.float32)

def load_lexicons_v3():
    """冻结词库 v3（work/lexicon/frozen_v3/lexicon_v3_frozen.json）。"""
    fr = json.load(open(LEX / "frozen_v3" / "lexicon_v3_frozen.json", encoding="utf-8"))
    lex = {}
    for sysname, facets in fr.items():
        if sysname == "meta":
            continue
        lex[sysname] = {f: {"en": list(v.get("en", [])), "zh": list(v.get("zh", []))}
                        for f, v in facets.items()}
    return lex


def load_lexicons():
    """返回 {system: {facet: {'en': [...], 'zh': [...]}}}。
    英文锚定词在 *_anchor*.json，中文翻译在 *_translated*.json 的 'zh' 键下。"""
    lex = {}
    # Hofstede: en = hofstede_anchor.json[ pole ].anchor_en；zh = verified_e5 翻译
    ha = json.load(open(LEX / "hofstede_anchor.json", encoding="utf-8"))
    ht = json.load(open(LEX / "hofstede_v2_translated_verified_e5.json", encoding="utf-8"))
    lex["hofstede"] = {}
    for pole, rec in ha.items():
        if pole == "meta":
            continue
        lex["hofstede"][pole] = {"en": list(rec.get("anchor_en", [])),
                                 "zh": list(ht["zh"].get(pole, []))}
    # WVS
    we = json.load(open(LEX / "wvs_anchor_en.json", encoding="utf-8"))
    wt = json.load(open(LEX / "wvs_anchor_translated_raw.json", encoding="utf-8"))
    lex["wvs"] = {}
    for facet, words in we.items():
        if facet == "meta":
            continue
        lex["wvs"][facet] = {"en": list(words),
                             "zh": list(wt["zh"].get(facet, []))}
    # Schwartz
    se = json.load(open(LEX / "schwartz_anchor_en.json", encoding="utf-8"))
    st = json.load(open(LEX / "schwartz_anchor_translated_raw.json", encoding="utf-8"))
    lex["schwartz"] = {}
    for facet, words in se.items():
        if facet == "meta":
            continue
        lex["schwartz"][facet] = {"en": list(words),
                                  "zh": list(st["zh"].get(facet, []))}
    # 社会主义核心价值观
    scv = json.load(open(LEX / "socialist_core_values.json", encoding="utf-8"))
    lex["scv"] = {}
    for k, v in scv.items():
        if k == "meta":
            continue
        lex["scv"][k] = {"en": v.get("anchor_en", []) or [v.get("en", "")],
                         "zh": v.get("anchor_zh", [])}
    return lex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    help="analysis_sets 下的文件名（不含 .parquet），或 corpora 名")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--max-texts", type=int, default=0, help="0=全部")
    ap.add_argument("--device", default="cpu",
                    help="默认 cpu。GPU 散热异常期间禁止使用；确需 GPU 须同时给 "
                         "--allow-gpu 且温度 < 70°C")
    ap.add_argument("--allow-gpu", action="store_true", default=False,
                    help="显式允许 GPU 计算（需人工确认散热已修复）")
    ap.add_argument("--lexicon", choices=["v1", "v3"], default="v1",
                    help="v1=锚定层（默认）；v3=冻结词库（LLM扩充+机器筛查）")
    ap.add_argument("--out-suffix", default="",
                    help="输出文件名后缀，如 _v3（默认按词库版本自动加）")
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程上限")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="半精度推理（默认开，显著降功耗）")
    ap.add_argument("--temp-limit", type=float, default=74.0,
                    help="超过此温度暂停散热（默认74°C）")
    ap.add_argument("--temp-resume", type=float, default=68.0,
                    help="散热至此温度以下继续（默认68°C）")
    ap.add_argument("--temp-abort", type=float, default=80.0,
                    help="超过此温度立即中止（默认80°C，低于83°C红线3°C）")
    ap.add_argument("--micro", type=int, default=256,
                    help="微批文本数（默认256，控制单批温升）")
    ap.add_argument("--doc-chunk-mean", action="store_true", default=False,
                    help="长文档分块均值嵌入（超长文本分块后加权均值，修复截断信息损失）")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large",
                    help="嵌入模型（默认 multilingual-e5-large；跨模型互证用 sentence-transformers/LaBSE 等）")
    ap.add_argument("--no-prefix", action="store_true", default=False,
                    help="不使用 query:/passage: 前缀（LaBSE 等不需要前缀的模型用）")
    args = ap.parse_args()

    if args.device == "cuda":
        t0 = gpu_temp()
        if not args.allow_gpu:
            raise SystemExit(
                f"拒绝启动：GPU 默认禁用（当前温度 {t0:.0f}°C）。"
                "确需 GPU 请显式加 --allow-gpu，并确保散热已维护。")
        if t0 > 70:
            raise SystemExit(f"拒绝启动：GPU 初始温度 {t0:.0f}°C > 70°C，先散热。")
    else:
        import torch
        torch.set_num_threads(args.threads)
        print(f"CPU 模式：线程上限 {args.threads}，GPU 全程不参与计算")

    from sentence_transformers import SentenceTransformer
    mk = {"torch_dtype": "float16"} if (args.fp16 and args.device == "cuda") else {}
    model = SentenceTransformer(args.model, device=args.device, model_kwargs=mk)
    t0 = gpu_temp()
    print(f"GPU 初始温度: {t0:.0f}°C  (限温 {args.temp_limit:.0f} / "
          f"恢复 {args.temp_resume:.0f} / 中止 {args.temp_abort:.0f})")

    # ---------- 刻面质心 ----------
    lex = load_lexicons_v3() if args.lexicon == "v3" else load_lexicons()
    n_words = sum(len([x for x in (w["en"] + w["zh"]) if x and len(x) < 40])
                  for facets in lex.values() for w in facets.values())
    print(f"词库版本: {args.lexicon}（{len(lex)} 体系, 约 {n_words} 词）")
    centroids = {}  # (system, facet) -> vec
    for sysname, facets in lex.items():
        for facet, w in facets.items():
            words = [x for x in (w["en"] + w["zh"]) if x and len(x) < 40]
            if not words:
                continue
            v = model.encode([("" if args.no_prefix else "query: ") + x for x in words],
                             normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=256, show_progress_bar=False)
            c = v.mean(axis=0)
            centroids[(sysname, facet)] = c / np.linalg.norm(c)
        print(f"刻面质心: {sysname} {len(facets)} 个")

    # ---------- 语料 ----------
    path = CORPUS / "analysis_sets" / f"{args.set}.parquet"
    if not path.exists():
        path = CORPUS / args.set / f"{args.set}.parquet"
    df = pd.read_parquet(path)
    if args.max_texts:
        df = df.head(args.max_texts)
    print(f"语料: {path.name}  {len(df):,} 行  culture={df['culture'].value_counts().to_dict()}")
    # 模型标签（区分缓存，避免 e5 与 LaBSE 等互相覆盖）
    mtag = "e5" if "e5" in args.model.lower() else args.model.split("/")[-1].lower().replace("-", "_")[:12]
    pfx = "" if args.no_prefix else "passage: "

    # ---------- 文本嵌入（分块缓存 + 温控 + 断点续跑） ----------
    cache_f = CACHE / f"{args.set}{'_cm' if args.doc_chunk_mean else ''}_{mtag}.npy"
    id_f = CACHE / f"{args.set}{'_cm' if args.doc_chunk_mean else ''}_{mtag}_ids.txt"

    if args.doc_chunk_mean:
        # 长文档分块：超过 max_chars 的文本按段落切成 ≤max_chars 的块，
        # 每块单独嵌入后按块长加权均值（再归一化）→ 文档向量。
        doc_ids, chunk_texts, chunk_w, doc_ptr = [], [], [], []
        for i, t in enumerate(df["text"].astype(str)):
            t = " ".join(t.split())
            if len(t) <= args.max_chars:
                doc_ptr.append((len(chunk_texts), 1, float(len(t))))
                chunk_texts.append(pfx + t)
                continue
            # 按段落优先切分
            paras = [p for p in re.split(r"(?<=[。！？.!?])\s*|\n+", t) if p.strip()]
            chunks, cur = [], ""
            for p in paras:
                if len(cur) + len(p) <= args.max_chars:
                    cur += p
                else:
                    if cur:
                        chunks.append(cur)
                    cur = p[: args.max_chars]
            if cur:
                chunks.append(cur)
            doc_ptr.append((len(chunk_texts), len(chunks), None))
            chunk_texts += [pfx + c for c in chunks]
        print(f"  分块均值嵌入: {len(df):,} 文档 -> {len(chunk_texts):,} 块")

        if cache_f.exists() and id_f.exists() and \
                id_f.read_text(encoding="utf-8").splitlines() == df["corpus_id"].tolist():
            emb = np.load(cache_f)
            print("使用嵌入缓存（分块均值）")
        else:
            def chunk_save(ci):
                return CACHE / f"{args.set}_cm_{mtag}_chunk{ci:03d}.npy"
            emb_chunks = encode_guarded(model, chunk_texts, args.batch,
                                        args.temp_limit, args.temp_resume, args.temp_abort,
                                        chunk_save=chunk_save, micro=args.micro)
            # 加权均值回文档向量
            emb = np.zeros((len(df), emb_chunks.shape[1]), dtype=np.float32)
            for i, (s, k, w) in enumerate(doc_ptr):
                blk = emb_chunks[s:s + k].astype(np.float32)
                if k == 1:
                    v = blk[0]
                else:
                    wts = np.array([max(1, len(chunk_texts[j]) - 9) for j in range(s, s + k)],
                                   dtype=np.float32)
                    v = (blk * wts[:, None]).sum(axis=0) / wts.sum()
                nrm = np.linalg.norm(v)
                emb[i] = v / nrm if nrm > 0 else v
            np.save(cache_f, emb.astype(np.float16))
            id_f.write_text("\n".join(df["corpus_id"].tolist()), encoding="utf-8")
            for ci in range((len(chunk_texts) + 1999) // 2000):
                chunk_save(ci).unlink(missing_ok=True)
    else:
        texts = [pfx + t[: args.max_chars] for t in df["text"]]
        if cache_f.exists() and id_f.exists() and \
                id_f.read_text(encoding="utf-8").splitlines() == df["corpus_id"].tolist():
            emb = np.load(cache_f)
            print("使用嵌入缓存")
        else:
            def chunk_save(ci):
                return CACHE / f"{args.set}_{mtag}_chunk{ci:03d}.npy"
            emb = encode_guarded(model, texts, args.batch,
                                 args.temp_limit, args.temp_resume, args.temp_abort,
                                 chunk_save=chunk_save, micro=args.micro)
            np.save(cache_f, emb.astype(np.float16))
            id_f.write_text("\n".join(df["corpus_id"].tolist()), encoding="utf-8")
            for ci in range((len(texts) + 1999) // 2000):
                chunk_save(ci).unlink(missing_ok=True)
    print(f"嵌入完成: {emb.shape}  GPU 温度: {gpu_temp():.0f}°C")

    # ---------- 相似度长表 ----------
    systems = sorted({s for s, _ in centroids})
    rows = []
    emb32 = emb.astype(np.float32)
    for (sysname, facet), c in centroids.items():
        sim = emb32 @ c
        rows.append(pd.DataFrame({
            "corpus_id": df["corpus_id"], "system": sysname,
            "facet": facet, "sim": sim}))
    sims = pd.concat(rows, ignore_index=True)
    meta_cols = [c for c in ["corpus_id", "culture", "genre", "source", "channel",
                             "item", "date", "year", "matched_film", "title", "rating"]
                 if c in df.columns]
    sims = sims.merge(df[meta_cols], on="corpus_id", how="left")
    suffix = args.out_suffix or ("_v3" if args.lexicon == "v3" else "")
    out = OUT_DIR / f"{args.set}_facet_sims{suffix}.parquet"
    sims.to_parquet(out, index=False)
    print(f"-> {out}  rows={len(sims):,}")
    # 透视宽表（每文本每刻面一列）
    wide = sims.pivot_table(index="corpus_id", columns=["system", "facet"],
                            values="sim")
    wide.to_parquet(OUT_DIR / f"{args.set}_facet_sims{suffix}_wide.parquet")
    print(f"-> {OUT_DIR / (args.set + '_facet_sims' + suffix + '_wide.parquet')}  shape={wide.shape}")


if __name__ == "__main__":
    main()
