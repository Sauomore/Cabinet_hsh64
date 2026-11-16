# -*- coding: utf-8 -*-
"""
跨语言检索评测：中文查询能不能在 Hamming 近邻里找到对应英文词。

为什么需要这个：
    混合索引的整个意义就是「查中文能召回英文」。之前的验证只做到
    embedding 层面（bge-m3 把「苹果」和「apple」放得近，AUC=1.0），
    那是**前提**，不是结论 —— 真正要证明的是**压缩成 52 位码之后**
    这个跨语言邻居关系还在。

指标：
    rank  = 目标英文词在「全部 6,109 个码」按 Hamming 距离排序中的位次
    top-k = rank <= k 的比例
    MRR   = 平均 1/rank
    机会基线 = 随机码的期望 rank ≈ N/2 = 3054

对照三种码：
    teacher   教师（bge-m3）余弦 —— 上限参考
    裸码      DeepHash 直接投影
    后处理    加贪心比特翻转之后

用法：
    python scripts/63_crosslingual_code_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval_helpers import load_exported  # noqa: E402
from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402
from cabinet_bnn.hsh.deep_hash import project, quantize_bits  # noqa: E402
from cabinet_bnn.hsh.post_optimize import load_override_cache  # noqa: E402

#: 与 scripts/43_probe_crosslingual.py 同一组人工词对
PAIRS = [
    ("苹果", "apple"), ("香蕉", "banana"), ("猫", "cat"), ("狗", "dog"),
    ("水", "water"), ("火", "fire"), ("太阳", "sun"), ("月亮", "moon"),
    ("山", "mountain"), ("河", "river"), ("树", "tree"), ("花", "flower"),
    ("学校", "school"), ("老师", "teacher"), ("学生", "student"),
    ("医生", "doctor"), ("医生", "physician"), ("书", "book"),
    ("计算机", "computer"), ("手机", "phone"), ("汽车", "car"),
    ("飞机", "airplane"), ("火车", "train"), ("北京", "Beijing"),
    ("上海", "Shanghai"), ("中国", "China"), ("美国", "America"),
    ("爱", "love"), ("恨", "hate"), ("快乐", "happy"), ("悲伤", "sad"),
    ("大", "big"), ("小", "small"), ("快", "fast"), ("慢", "slow"),
    ("热", "hot"), ("冷", "cold"), ("新", "new"), ("旧", "old"),
]

NEG_PAIRS = [
    ("苹果", "mountain"), ("猫", "computer"), ("水", "teacher"),
    ("太阳", "sad"), ("书", "airplane"), ("爱", "stone"),
]


def hamming_row(query_bits: np.ndarray, all_bits: np.ndarray) -> np.ndarray:
    """单个查询码到全部码的 Hamming 距离。"""
    return (all_bits != query_bits[None, :]).sum(axis=1).astype(np.int32)


def rank_of(target: int, dist: np.ndarray, self_idx: int) -> int:
    """target 在按距离升序排列中的位次（排除自身，1-based）。"""
    order = np.argsort(dist, kind="stable")
    order = order[order != self_idx]
    pos = np.where(order == target)[0]
    return int(pos[0]) + 1 if len(pos) else len(order) + 1


def summarise(ranks: list[int], name: str) -> dict:
    r = np.array(ranks, dtype=np.float64)
    out = {
        "name": name, "n": int(len(r)),
        "median_rank": float(np.median(r)),
        "mrr": float(np.mean(1.0 / r)),
        "top1": float((r <= 1).mean()),
        "top5": float((r <= 5).mean()),
        "top10": float((r <= 10).mean()),
        "top50": float((r <= 50).mean()),
        "worst": int(r.max()),
    }
    print(f"  {name:<12} n={out['n']:<3} 中位 rank {out['median_rank']:>7.1f}  "
          f"MRR {out['mrr']:.4f}  top1 {out['top1']:.1%}  top5 {out['top5']:.1%}  "
          f"top10 {out['top10']:.1%}  最差 {out['worst']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="跨语言码检索评测")
    ap.add_argument("--vocab", type=Path, default=ROOT / "data/mixed_vocab.txt")
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "data/mixed_embedding.cache")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "data/mixed_deep_hash_52.bin")
    ap.add_argument("--override", type=Path,
                    default=ROOT / "data/eval/sim_override_mixed6109.bin")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/crosslingual_code_eval.json")
    args = ap.parse_args()

    words = [l.strip() for l in
             args.vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = {w: i for i, w in enumerate(words)}
    n = len(words)
    print("=" * 78)
    print(f"跨语言码检索评测   {n:,} 词混合词表")
    print("=" * 78)

    # 教师（bge-m3）
    dim, cw, teacher = read_cache(args.cache)
    if cw != words:
        perm = [ {w: i for i, w in enumerate(cw)}[w] for w in words ]
        teacher = teacher[perm]
    tn = normalize_rows(teacher)
    print(f"  教师 {dim} 维")

    # 裸码
    mean, model = load_exported(args.model)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    raw_bits = quantize_bits(project(model, mean, teacher, device=dev))

    # 后处理码
    if args.override.exists():
        ow, obits = load_override_cache(args.override)
        oi = {w: i for i, w in enumerate(ow)}
        post_bits = np.zeros_like(raw_bits)
        for i, w in enumerate(words):
            if w in oi:
                post_bits[i] = obits[oi[w]]
        has_post = True
    else:
        post_bits = raw_bits
        has_post = False
        print("  ⚠️ 没有后处理覆盖码，只评裸码")

    # 可配对的词对（两边都在词表里）
    usable = [(z, e) for z, e in PAIRS if z in idx and e in idx]
    missing = [(z, e) for z, e in PAIRS if z not in idx or e not in idx]
    print(f"  词对：{len(usable)} 可用 / {len(missing)} 缺失（不在词表）")
    if missing:
        print(f"    缺失示例: {missing[:6]}")
    print()

    results = {}
    per_pair = []
    print("  中文 → 英文")
    for label, bits in (("裸码", raw_bits), ("后处理", post_bits) if has_post else ()):
        ranks = []
        for z, e in usable:
            d = hamming_row(bits[idx[z]], bits)
            r = rank_of(idx[e], d, idx[z])
            ranks.append(r)
            per_pair.append({"zh": z, "en": e, "metric": label, "rank": r})
        results[label] = summarise(ranks, label)

    # 逐条明细，方便看长尾失败的是哪些词对
    if has_post:
        post_ranks = {p["zh"] + "|" + p["en"]: p["rank"]
                      for p in per_pair if p["metric"] == "后处理"}
        print()
        print("  逐条（按 rank 降序，只看最差 10 条）:")
        worst = sorted(post_ranks.items(), key=lambda kv: -kv[1])[:10]
        for key, r in worst:
            z, e = key.split("|")
            print(f"    {z} → {e:<12} rank {r}")

    # 教师上限参考
    ranks_t = []
    for z, e in usable:
        sim = tn[idx[z]] @ tn.T
        sim[idx[z]] = -np.inf
        order = np.argsort(-sim)
        ranks_t.append(int(np.where(order == idx[e])[0][0]) + 1)
    results["teacher"] = summarise(ranks_t, "teacher")

    # 负对照
    if has_post:
        neg_usable = [(a, b) for a, b in NEG_PAIRS if a in idx and b in idx]
        if neg_usable:
            print()
            print("  负对照（本不该被召回）")
            ranks_neg = []
            for a, b in neg_usable:
                d = hamming_row(post_bits[idx[a]], post_bits)
                ranks_neg.append(rank_of(idx[b], d, idx[a]))
            results["neg_control"] = summarise(ranks_neg, "负对照")

    print()
    print("=" * 78)
    print("解读")
    print("=" * 78)
    if has_post:
        d = results["后处理"]
        chance = n / 2
        print(f"  机会基线（随机码）期望 rank ≈ {chance:,.0f}")
        print(f"  后处理码中位 rank {d['median_rank']:.1f}  →  比随机好 "
              f"{chance / max(d['median_rank'],1):,.0f} 倍")
        t = results["teacher"]
        print(f"  教师中位 rank {t['median_rank']:.1f}  →  码相对教师损失 "
              f"{d['median_rank'] - t['median_rank']:+.1f} 位")
        print(f"  top-10 命中率：裸码 {results['裸码']['top10']:.1%} → "
              f"后处理 {d['top10']:.1%}（教师 {t['top10']:.1%}）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_words": n, "n_pairs_usable": len(usable),
        "missing_pairs": [list(p) for p in missing],
        "results": results,
        "per_pair": per_pair,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
