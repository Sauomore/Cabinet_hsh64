# -*- coding: utf-8 -*-
"""
测 Hamming 距离分布：正样本对 vs 负样本对。

为什么需要：
    MIH 索引的 radius 该设多大，取决于正样本对的真实 Hamming 距离。
    benchmark 只试了 radius <= 8，而如果正样本平均距离是 9-10，
    radius=8 就一个都抓不到 —— 这解释了「平均候选 2.3 / 召回 0.13」。

同时给出论文图 2 的数值版：随机基线应为 26（52/2）。

用法：
    python scripts/55_distance_stats.py --vocab tests/data/vocab_3109.txt \
        --model F:/Cabinet-BNN/results/deep_hash_mvp1_h512_s42.bin \
        --override F:/Cabinet-BNN/results/sim_override_mvp1_h512_s42.bin \
        --teacher-cache F:/Cabinet-BNN/results/bge_large_3109.cache
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402
from cabinet_bnn.hsh.deep_hash import bits_to_u64, project, quantize_bits  # noqa: E402
from cabinet_bnn.hsh.post_optimize import load_override_cache  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from _eval_helpers import load_exported, load_pca_student  # noqa: E402


def hamming_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,52) 位矩阵 -> (N,N) Hamming 距离（float32，分块）。"""
    n = a.shape[0]
    out = np.empty((n, n), dtype=np.float32)
    blk = max(1, int(4_000_000 // max(n, 1)))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        out[s:e] = (a[s:e, None, :] != b[None, :, :]).sum(axis=2, dtype=np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Hamming 距离分布")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--override", type=Path, default=None)
    ap.add_argument("--pca-basis", type=Path, default=None)
    ap.add_argument("--student-dim", type=int, default=512)
    ap.add_argument("--pos-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    words = [l.strip() for l in
             args.vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(words)
    print("=" * 72)
    print(f"Hamming 距离分布  {args.vocab.name}  {n:,} 词")
    print("=" * 72)

    # 教师
    dim_t, tw, teacher = read_cache(args.teacher_cache)
    if tw != words:
        idx = {w: i for i, w in enumerate(tw)}
        teacher = teacher[[idx[w] for w in words]]
    tn = normalize_rows(teacher)
    ts = tn @ tn.T
    print(f"  教师 {dim_t} 维")

    # 学生（PCA-512）
    student = load_pca_student(teacher, args.student_dim, args.pca_basis)
    print(f"  学生 {student.shape}")

    # 模型
    mean, model = load_exported(args.model)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    bits = quantize_bits(project(model, mean, student, device=dev))
    print(f"  模型 -> 唯一 sim 码 {len(np.unique(bits_to_u64(bits))):,} / {n:,}")

    def stats(b, name):
        d = hamming_pairs(b, b)
        # 正样本：teacher 的 top-10（排除自身）
        pos_idx = np.argsort(-ts, axis=1)[:, 1:args.pos_k + 1]
        pos_d = np.take_along_axis(d, pos_idx, axis=1)
        # 负样本：teacher 的最远 10
        neg_idx = np.argsort(ts, axis=1)[:, :args.pos_k]
        neg_d = np.take_along_axis(d, neg_idx, axis=1)
        # 随机基线
        rng = np.random.default_rng(0)
        ri = rng.integers(0, n, size=(n, args.pos_k))
        rand_d = np.take_along_axis(d, ri, axis=1)
        r = {
            "name": name,
            "pos_mean": float(pos_d.mean()), "pos_std": float(pos_d.std()),
            "pos_p5": float(np.percentile(pos_d, 5)),
            "pos_p50": float(np.percentile(pos_d, 50)),
            "pos_p95": float(np.percentile(pos_d, 95)),
            "neg_mean": float(neg_d.mean()), "neg_std": float(neg_d.std()),
            "rand_mean": float(rand_d.mean()), "rand_std": float(rand_d.std()),
            "separation": float(neg_d.mean() - pos_d.mean()),
            "m": int(b.shape[1]),
        }
        print(f"\n  [{name}]  {r['m']} bit")
        print(f"    正样本  均值 {r['pos_mean']:6.2f}  std {r['pos_std']:5.2f}  "
              f"P5 {r['pos_p5']:5.1f}  P50 {r['pos_p50']:5.1f}  P95 {r['pos_p95']:5.1f}")
        print(f"    负样本  均值 {r['neg_mean']:6.2f}  std {r['neg_std']:5.2f}")
        print(f"    随机    均值 {r['rand_mean']:6.2f}  std {r['rand_std']:5.2f}  "
              f"(理论 {r['m']/2:.1f})")
        print(f"    分离度  {r['separation']:6.2f} bit")
        # 覆盖正样本所需的 radius
        for pct in (50, 80, 90, 95):
            need = int(np.ceil(np.percentile(pos_d, pct)))
            print(f"      覆盖 {pct}% 正样本需要 radius >= {need}")
        return r

    results = [stats(bits, "DeepHash 裸码")]

    if args.override and args.override.exists():
        ow, obits = load_override_cache(args.override)
        idx = {w: i for i, w in enumerate(ow)}
        opt = np.zeros_like(bits)
        for i, w in enumerate(words):
            if w in idx:
                opt[i] = obits[idx[w]]
        results.append(stats(opt, "后处理后"))

    out = args.out or (ROOT / "results/distance_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
    key = f"{args.vocab.stem}|{n}"
    prev[key] = results
    out.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
