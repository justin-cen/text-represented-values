# -*- coding: utf-8 -*-
"""模块E：multilingual-e5-large 中英跨语言对齐质量校验（方法章节证据）。

检验1（词级检索）：MUSE en-zh 真值词对，top-1/top-5 检索准确率 + 真对 vs 随机对余弦分离度。
检验2（句级判别）：中英平行句对应高于非平行句。

输出: docs/e5_alignment_check.txt
"""
import io
import random
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(r"E:\culture-difference")
OUT = BASE / "data_corpus" / "docs" / "e5_alignment_check.txt"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

N_PAIRS = 2000
SEED = 42


def load_muse(path, n):
    pairs, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            en, zh = parts[0], parts[1]
            if en.isalpha() and en not in seen:
                seen.add(en)
                pairs.append((en.lower(), zh))
            if len(pairs) >= n:
                break
    return pairs


def main():
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")

    def emb(texts):
        return model.encode(["query: " + t for t in texts], normalize_embeddings=True,
                            convert_to_numpy=True, batch_size=256, show_progress_bar=False)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    pairs = load_muse(BASE / "work" / "data" / "muse" / "en-zh.txt", N_PAIRS)
    log(f"MUSE 词对数: {len(pairs)}")
    en = emb([p[0] for p in pairs])
    zh = emb([p[1] for p in pairs])

    # 真对 vs 随机对
    rng = np.random.RandomState(SEED)
    true_sim = np.sum(en * zh, axis=1)
    perm = rng.permutation(len(pairs))
    rand_sim = np.sum(en * zh[perm], axis=1)
    log(f"真对余弦:  mean={true_sim.mean():.4f}  p50={np.median(true_sim):.4f}")
    log(f"随机对余弦: mean={rand_sim.mean():.4f}  p50={np.median(rand_sim):.4f}")
    log(f"分离度 (真-随机均值差): {true_sim.mean() - rand_sim.mean():.4f}")
    auc = (true_sim[:, None] > rand_sim[None, :]).mean()
    log(f"AUC(真对>随机对): {auc:.4f}")

    # top-k 检索准确率
    sims = en @ zh.T  # [N, N]
    top1 = (sims.argmax(axis=1) == np.arange(len(pairs))).mean()
    top5 = np.any(np.argsort(-sims, axis=1)[:, :5] == np.arange(len(pairs))[:, None], axis=1).mean()
    log(f"Top-1 准确率: {top1:.4f}   Top-5 准确率: {top5:.4f}")

    # 句级
    para = [
        ("We must work together to address global challenges.", "我们必须携手应对全球性挑战。"),
        ("Individual freedom is the foundation of creativity.", "个人自由是创造力的根基。"),
        ("Family responsibility comes before personal wishes.", "家庭责任重于个人意愿。"),
        ("The government announced new economic reforms.", "政府宣布了新的经济改革措施。"),
    ]
    all_en = emb([p[0] for p in para])
    all_zh = emb([p[1] for p in para])
    m = all_en @ all_zh.T
    log("\n句级平行判别（对角线应为行内最大）:")
    for i, (e, z) in enumerate(para):
        ok = "✓" if m[i].argmax() == i else "✗"
        log(f"  {ok} en[{i}]·zh[{i}]={m[i, i]:.3f}  行内最大={m[i].max():.3f}  {e[:40]}")
    log(f"\n结论: multilingual-e5-large 中英对齐质量"
        f"{'良好，可用于统一空间中西文本嵌入' if top5 > 0.7 and m.diagonal().mean() > m.max(axis=1).mean() - 0.05 else '一般，需谨慎'}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
