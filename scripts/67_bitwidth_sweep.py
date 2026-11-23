# -*- coding: utf-8 -*-
"""
码宽扫描：定位 DeepHash 相对 PCA+sign 的优势区间。

背景
----
单种子、3 档位宽的实验显示一个反常模式：
    裸码  20 位：DeepHash 0.4040  >>  PCA 0.1537   （DH 大胜）
    裸码  32 位：DeepHash 0.4453  >   PCA 0.3600   （DH 胜）
    裸码  52 位：DeepHash 0.4979  <   PCA 0.6125   （PCA 反超）
说明 DH 的价值集中在「位宽不足」时。本脚本用更密的位宽网格与多种子把这条曲线画清楚，
回答两个问题：
    ① 交叉点在哪一档位宽？
    ② DeepHash 在低位宽的优势是否稳定（跨种子）？

为什么这件事对论文重要
----------------------
若「短码下学习是必需的」成立，DeepHash 就有了明确且可辩护的适用边界，
而不是「在 52 位下不如 PCA」这种尴尬结论。

设计
----
- 位宽：8, 12, 16, 20, 32, 40, 48, 52
- 种子：42, 2024, 2025（PC1 线性基线本身确定性，不重复）
- 其余固定：隐藏维度 512、epochs 500、pos_k=10、后处理 pos_k=10/neg_weight=1.0/max_iters=10
- 指标：overlap@10（论文口径），并记录唯一码率

用法
----
    python scripts/67_bitwidth_sweep.py --tag synth3109 \
        --teacher F:/Cabinet-BNN/results/bge_large_3109.cache \
        --student F:/HSH_64_dev/data/eval/student_repro3109_d512.cache
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

import torch  # noqa: E402
from cabinet_bnn.hsh.deep_hash import (  # noqa: E402
    bits_to_u64, project, quantize_bits, train_deep_hash,
)
from cabinet_bnn.hsh.post_optimize import greedy_flip_optimize, hamming_matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EVAL = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED
DEFAULT_WIDTHS = [8, 12, 16, 20, 32, 40, 48, 52]
DEFAULT_SEEDS = [42, 2024, 2025]


def read_cache(path: Path) -> tuple[int, list[str], np.ndarray]:
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


def recall_with_dist(pos: np.ndarray, dist: np.ndarray, k: int) -> float:
    n = dist.shape[0]
    kk = min(k, n - 1)
    total, counted = 0.0, 0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        hit = np.intersect1d(pos[i], order[:kk], assume_unique=False).size
        total += hit / max(pos.shape[1], 1)
        counted += 1
    return total / counted if counted else 0.0


def eval_bits(bits: np.ndarray, sim: np.ndarray, k: int = 10) -> float:
    d = hamming_matrix(bits)
    kk = min(k, sim.shape[0] - 1)
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1 : kk + 1]
    return recall_with_dist(pos, d, kk)


def cosine_sim(x: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return xn @ xn.T


def pca_sign(teacher: np.ndarray, width: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    p = PCA(n_components=width, random_state=0)
    return (p.fit_transform(teacher) > 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="synth3109")
    ap.add_argument("--teacher", type=Path, default=None)
    ap.add_argument("--student", type=Path, default=None)
    ap.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", type=Path, default=RESULTS / "bitwidth_sweep.json")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    tag = args.tag
    tpath = args.teacher or (EVAL / f"teacher_{tag}.cache")
    spath = args.student or (EVAL / f"student_{tag}_d512.cache")
    if not tpath.exists() or not spath.exists():
        print(f"  ❌ 缺缓存：{tpath} / {spath}")
        return 1

    print("=" * 84)
    print(f"  码宽扫描 —— {tag}   位宽 {widths}   种子 {seeds}")
    print("=" * 84)

    dim_t, _, teacher = read_cache(tpath)
    _, _, student = read_cache(spath)
    n = teacher.shape[0]
    teacher_sim = cosine_sim(teacher)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  教师 {dim_t}d / 学生 {student.shape[1]}d / {n} 词 / 设备 {dev}")

    out = {"tag": tag, "n": n, "widths": widths, "seeds": seeds, "results": {}}
    t_start = time.time()

    for width in widths:
        wt0 = time.time()
        r = {}

        # ---------- PCA+sign（确定性基线）
        bits_pca = pca_sign(teacher, width)
        r["pca_bare"] = eval_bits(bits_pca, teacher_sim)
        bits_pca_opt = greedy_flip_optimize(
            bits_pca, teacher_sim, pos_k=10, k=10,
            neg_weight=1.0, max_iters=10, verbose=False)
        r["pca_post"] = eval_bits(bits_pca_opt, teacher_sim)

        # ---------- DeepHash × 多个种子
        bare, post, uniq = [], [], []
        for seed in seeds:
            mean, model, _h = train_deep_hash(
                student, n_bits=width, hidden_dim=args.hidden_dim,
                epochs=args.epochs, lr=1e-3, pos_k=10,
                teacher_x=teacher, seed=seed, device=dev,
                log_every=0, verbose=False)
            u = project(model, mean, student, device=dev)
            bits = quantize_bits(u)
            uniq.append(len(np.unique(bits_to_u64(bits))) / n)
            bare.append(eval_bits(bits, teacher_sim))
            opt = greedy_flip_optimize(
                bits, teacher_sim, pos_k=10, k=10,
                neg_weight=1.0, max_iters=10, verbose=False)
            post.append(eval_bits(opt, teacher_sim))

        r["deephash_bare"] = {"mean": float(np.mean(bare)), "std": float(np.std(bare)),
                              "runs": bare}
        r["deephash_post"] = {"mean": float(np.mean(post)), "std": float(np.std(post)),
                              "runs": post}
        r["deephash_unique_rate"] = {"mean": float(np.mean(uniq)), "runs": uniq}
        out["results"][str(width)] = r

        dt = time.time() - wt0
        print(f"\n  {width:>3} 位  ({dt:.0f}s)")
        print(f"    PCA  裸码 {r['pca_bare']:.4f}   后处理 {r['pca_post']:.4f}")
        print(f"    DH   裸码 {r['deephash_bare']['mean']:.4f}±{r['deephash_bare']['std']:.4f}"
              f"   后处理 {r['deephash_post']['mean']:.4f}±{r['deephash_post']['std']:.4f}"
              f"   唯一码 {r['deephash_unique_rate']['mean']:.2%}")

    # ---------- 汇总
    print("\n" + "=" * 84)
    print("  overlap@10 汇总（PCA 确定性；DH 为多种子均值±标准差）")
    print("=" * 84)
    hdr = "".join(f"{w:>12}" for w in widths)
    print(f"    {'方法':<16}{hdr}")
    rows = [
        ("PCA 裸码", lambda r: f"{r['pca_bare']:.4f}"),
        ("PCA + 后处理", lambda r: f"{r['pca_post']:.4f}"),
        ("DH 裸码", lambda r: f"{r['deephash_bare']['mean']:.4f}"),
        ("DH + 后处理", lambda r: f"{r['deephash_post']['mean']:.4f}"),
    ]
    for name, f in rows:
        print(f"    {name:<16}" + "".join(f"{f(out['results'][str(w)]):>12}" for w in widths))

    # ---------- 交叉点
    def crossover(key_pca, key_dh):
        prev = None
        for w in widths:
            r = out["results"][str(w)]
            pv = r[key_pca]
            dv = r[key_dh]["mean"]
            if prev is not None and pv > dv and prev[1] <= prev[0]:
                return prev[2], w
            prev = (dv, pv, w)
        return None

    print()
    cx_bare = crossover("pca_bare", "deephash_bare")
    cx_post = crossover("pca_post", "deephash_post")
    out["crossover_bare"] = cx_bare
    out["crossover_post"] = cx_post
    if cx_bare:
        print(f"  裸码交叉点：DH 在 {cx_bare[0]} 位领先，到 {cx_bare[1]} 位被 PCA 反超")
    if cx_post:
        print(f"  后处理后交叉点：{cx_post[0]} 位 → {cx_post[1]} 位之间")
    if not cx_bare and not cx_post:
        print("  在扫描区间内未观察到交叉（PCA 全程领先或全程落后）")

    out["seconds"] = time.time() - t_start
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
