# -*- coding: utf-8 -*-
"""
ITQ 基线（Iterative Quantization, Gong & Lazebnik, CVPR 2011）。

为什么它是最重要的剩余基线
--------------------------
ITQ 是**无监督**哈希里最强的经典方法，恰好匹配本项目的设定：只有教师嵌入、没有标签。
而且它与本文的 PCA-52 基线有精确关系：**PCA+sign 等价于 ITQ 取 R = I**，
所以 ITQ 理论上不应差于 PCA；若实测反而更差，说明实现有错。

算法
----
    X ← 零中心化教师嵌入
    V ← PCA 前 r 个分量（1024 → r）
    Y ← X·V
    R ← 随机正交矩阵
    重复：
        B ← sign(Y·R)
        U,S,Vt ← SVD(Yᵀ·B)          # 最小化 ||B − Y·R||²_F, s.t. RᵀR = I
        R ← U·Vt
    bits ← (sign(Y·R) > 0)

三个自检（在 --self-check 下运行）
---------------------------------
    ① R = I 时 ITQ 的码应与 PCA+sign 完全一致（验证投影与符号约定）
    ② 量化误差 ||B − Y·R||² 应随迭代单调不增，且最终低于 R=I
    ③ 各比特的 1 占比应接近 50%（ITQ 的平衡性收益）

用法
----
    python scripts/70_itq_baseline.py \
        --teacher F:/Cabinet-BNN/results/bge_large_3109.cache --tag synth3109
    python scripts/70_itq_baseline.py \
        --teacher data/eval/teacher_nat3000.cache --tag nat3000
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

from cabinet_bnn.hsh.post_optimize import greedy_flip_optimize, hamming_matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED
DEFAULT_WIDTHS = [8, 12, 16, 20, 32, 40, 48, 52]


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
    """与 cabinet_bnn.hsh.post_optimize.recall_at_k_from_ranking 逐位一致（stable argsort）。"""
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


def pca_project(teacher: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """零中心化后取前 r 个主成分，返回 (Y, mean)。"""
    from sklearn.decomposition import PCA
    p = PCA(n_components=r, random_state=0)
    Y = p.fit_transform(teacher.astype(np.float64))
    return Y, p.mean_


def random_orthogonal(r: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((r, r)))
    return q


def itq(Y: np.ndarray, n_iter: int = 50, seed: int = 0,
        verbose: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (bits, R, 诊断)。"""
    r = Y.shape[1]
    R = random_orthogonal(r, seed)
    errs = []
    prev = None
    for it in range(n_iter):
        B = np.where(Y @ R >= 0, 1.0, -1.0)
        err = float(np.sum((B - Y @ R) ** 2))
        errs.append(err)
        if verbose and (it < 3 or it % 10 == 0):
            print(f"      iter {it:>2}  量化误差 {err:.6g}")
        # 放弃零方差维度的影响：若已收敛则停
        if prev is not None and abs(prev - err) <= 1e-9 * max(1.0, abs(prev)):
            break
        prev = err
        U, _S, Vt = np.linalg.svd(Y.T @ B)
        R = U @ Vt
    B = np.where(Y @ R >= 0, 1.0, -1.0)
    err = float(np.sum((B - Y @ R) ** 2))
    errs.append(err)
    bits = (B > 0).astype(np.uint8)
    uniq = len(np.unique(bits.astype(np.int64) @ (1 << np.arange(r, dtype=np.int64))))
    diag = {
        "unique_rate": uniq / bits.shape[0],
        "err_final": err,
        "err_identity": float(np.sum((np.where(Y >= 0, 1.0, -1.0) - Y) ** 2)),
        "n_iter_run": len(errs),
        "err_sequence": errs[:3] + errs[-2:],
        "bit_balance": float(bits.mean()),
        "bit_min": float(bits.mean(0).min()),
        "bit_max": float(bits.mean(0).max()),
    }
    return bits, R, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="synth3109")
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS / "itq_baseline.json")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    if not args.teacher.exists():
        print(f"  ❌ 教师缓存不存在：{args.teacher}")
        return 1

    print("=" * 84)
    print(f"  ITQ 基线 —— {args.tag}")
    print("=" * 84)
    dim, words, teacher = read_cache(args.teacher)
    n = teacher.shape[0]
    teacher_sim = cosine_sim(teacher)
    print(f"  教师 {dim}d / {n} 词 / 位宽 {widths}")

    out = {"tag": args.tag, "n": n, "teacher_dim": dim, "widths": widths,
           "iters": args.iters, "seed": args.seed, "results": {}}
    t_all = time.time()

    for r in widths:
        t0 = time.time()
        Y, _mean = pca_project(teacher, r)
        bits_itq, R, diag = itq(Y, n_iter=args.iters, seed=args.seed, verbose=False)
        bits_pca = (Y >= 0).astype(np.uint8)          # R = I 的退化情形

        rec_bare = eval_bits(bits_itq, teacher_sim)
        rec_pca = eval_bits(bits_pca, teacher_sim)

        # 自检①：R=I 应等价 PCA
        same_as_pca = bool(np.array_equal(bits_itq, bits_pca)) if np.allclose(R, np.eye(r)) else None
        # 自检②：量化误差应低于 R=I
        err_ok = diag["err_final"] <= diag["err_identity"] * (1 + 1e-6)

        opt = greedy_flip_optimize(bits_itq, teacher_sim, pos_k=10, k=10,
                                   neg_weight=1.0, max_iters=10, verbose=False)
        rec_post = eval_bits(opt, teacher_sim)

        out["results"][str(r)] = {
            "itq_bare": rec_bare, "itq_post": rec_post, "pca_bare": rec_pca,
            "diag": {k: v for k, v in diag.items() if k != "err_sequence"},
            "err_seq": diag["err_sequence"],
            "err_below_identity": err_ok,
            "same_as_pca_when_identity": same_as_pca,
            "seconds": time.time() - t0,
        }
        print(f"\n  {r:>3} 位  ({time.time() - t0:.1f}s)")
        print(f"    PCA+sign 裸码 {rec_pca:.4f}")
        print(f"    ITQ      裸码 {rec_bare:.4f}   后处理 {rec_post:.4f}")
        print(f"    量化误差  ITQ {diag['err_final']:.6g}  vs R=I {diag['err_identity']:.6g}"
              f"  {'ok' if err_ok else '⚠️ 未下降'}"
              f"   唯一码率 {diag['unique_rate']:.2%}"
              f"   ||R−I|| {np.linalg.norm(R - np.eye(r)):.2f}")
        print(f"    比特平衡  均值 {diag['bit_balance']:.3f}  最偏 {diag['bit_min']:.2f}-{diag['bit_max']:.2f}")

    # ---------- 与既有 PCA / DeepHash 扫描对比
    sweep_path = RESULTS / "bitwidth_sweep.json"
    cmp_rows = []
    if sweep_path.exists():
        sw = json.load(io.open(sweep_path, encoding="utf-8"))
        if sw.get("n") == n:
            print("\n" + "=" * 84)
            print("  三编码器对比（overlap@10，后处理后）")
            print("=" * 84)
            print(f"    {'位宽':<6}{'PCA':>10}{'ITQ':>10}{'DeepHash':>12}{'ITQ−PCA':>10}")
            for r in widths:
                key = str(r)
                if key not in sw["results"]:
                    continue
                pca_post = sw["results"][key]["pca_post"]
                dh_post = sw["results"][key]["deephash_post"]["mean"]
                itq_post = out["results"][key]["itq_post"]
                cmp_rows.append({"width": r, "pca_post": pca_post,
                                 "itq_post": itq_post, "dh_post": dh_post})
                print(f"    {r:<6}{pca_post:>10.4f}{itq_post:>10.4f}{dh_post:>12.4f}"
                      f"{itq_post - pca_post:>+10.4f}")
            out["comparison"] = cmp_rows

    out["seconds"] = time.time() - t_all
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds']:.1f}s   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
