# -*- coding: utf-8 -*-
"""
C. 主结果表的 5 种子方差

问题
----
主结果表（h128 / h256 / h512 / h1024 × 有无后处理）仍是单一 seed。
码宽消融与 LSH/ITQ 已用 3 种子，主表没有 —— 这是最容易被审稿人指出的漏洞。

做法
----
四个隐藏维度 × 5 个种子（42 / 0 / 1 / 2024 / 2025），
每个配置报告裸码与后处理的均值 ± 标准差，并与原论文记录值对照。

用法
----
    python scripts/77_main_table_seeds.py --teacher F:/Cabinet-BNN/results/bge_large_3109.cache
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

from cabinet_bnn.hsh.deep_hash import project, quantize_bits, train_deep_hash  # noqa: E402
from cabinet_bnn.hsh.post_optimize import greedy_flip_optimize, hamming_matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EVAL = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED
HIDDENS = [128, 256, 512, 1024]
SEEDS = [42, 0, 1, 2024, 2025]
# 原论文记录值（单 seed），用于对照
RECORDED = {128: 0.4228, 256: 0.4315, 512: 0.4265, 1024: 0.4315}


def read_cache(path: Path):
    with path.open("rb") as f:
        magic, _v, dim, count = struct.unpack(">4I", f.read(16))
        if magic != CACHE_MAGIC:
            raise ValueError(f"{path.name}: magic 0x{magic:X}")
        words, vecs = [], np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            (blen,) = struct.unpack(">H", f.read(2))
            words.append(f.read(blen).decode("utf-8"))
            vecs[i] = np.frombuffer(f.read(dim * 4), dtype=">f4").astype(np.float32)
    return dim, words, vecs


def recall_with_dist(pos, dist, k):
    n = dist.shape[0]
    kk = min(k, n - 1)
    tot = 0.0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        tot += np.intersect1d(pos[i], order[:kk], assume_unique=False).size / pos.shape[1]
    return tot / n


def eval_bits(bits, pos, k=10):
    return recall_with_dist(pos, hamming_matrix(bits), k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--student", type=Path, default=EVAL / "student_repro3109_d512.cache")
    ap.add_argument("--bits", type=int, default=52)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", type=Path, default=RESULTS / "main_table_seeds.json")
    args = ap.parse_args()

    dim, words, teacher = read_cache(args.teacher)
    n = teacher.shape[0]
    tn = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)
    sim = tn @ tn.T
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    if not args.student.exists():
        print(f"  ❌ 缺学生缓存 {args.student}")
        return 1
    _, _, student = read_cache(args.student)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print(f"  C. 主结果表 5 种子方差 —— {n} 词 / {args.bits} 位 / {len(SEEDS)} 种子")
    print("=" * 92)

    out = {"n": n, "bits": args.bits, "seeds": SEEDS, "hiddens": HIDDENS, "results": {}}
    t_all = time.time()

    for h in HIDDENS:
        print(f"\n  ── h{h} ──")
        bare, post = [], []
        for sd in SEEDS:
            t0 = time.time()
            mean, model, _hh = train_deep_hash(
                student, n_bits=args.bits, hidden_dim=h, epochs=args.epochs, lr=1e-3,
                pos_k=10, teacher_x=teacher, seed=sd, device=dev, log_every=0, verbose=False)
            bits = quantize_bits(project(model, mean, student, device=dev))
            r_bare = eval_bits(bits, pos)
            opt = greedy_flip_optimize(bits, sim, pos_k=10, k=10, neg_weight=1.0,
                                       max_iters=10, verbose=False)
            r_post = eval_bits(opt, pos)
            bare.append(r_bare)
            post.append(r_post)
            print(f"    seed {sd:<5} 裸码 {r_bare:.4f}  后处理 {r_post:.4f}   ({time.time()-t0:.0f}s)")
        out["results"][str(h)] = {
            "bare": {"mean": float(np.mean(bare)), "std": float(np.std(bare)), "runs": bare},
            "post": {"mean": float(np.mean(post)), "std": float(np.std(post)), "runs": post},
            "recorded_bare": RECORDED[h],
        }
        print(f"    → 裸码 {np.mean(bare):.4f} ± {np.std(bare):.4f}"
              f"（记录 {RECORDED[h]:.4f}）   后处理 {np.mean(post):.4f} ± {np.std(post):.4f}")

    print("\n" + "=" * 92)
    print("  汇总（overlap@10，均值 ± 标准差）")
    print("=" * 92)
    print(f"    {'隐藏维度':<10}{'裸码':>18}{'记录值':>10}{'后处理':>18}")
    for h in HIDDENS:
        v = out["results"][str(h)]
        print(f"    h{h:<9}{v['bare']['mean']:>11.4f} ± {v['bare']['std']:.4f}"
              f"{v['recorded_bare']:>10.4f}"
              f"{v['post']['mean']:>13.4f} ± {v['post']['std']:.4f}")

    # 后处理之后的跨维度极差 vs 跨种子方差
    posts = [out["results"][str(h)]["post"]["mean"] for h in HIDDENS]
    stds = [out["results"][str(h)]["post"]["std"] for h in HIDDENS]
    spread = max(posts) - min(posts)
    mean_std = float(np.mean(stds))
    print(f"\n    后处理后果：跨隐藏维度极差 {spread:.4f}   跨种子平均标准差 {mean_std:.4f}")
    print(f"    → {'维度差异大于种子噪声' if spread > 2 * mean_std else '维度差异与种子噪声同量级（排序需谨慎）'}")
    out["spread_across_hidden"] = spread
    out["mean_seed_std"] = mean_std

    out["seconds"] = time.time() - t_all
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
