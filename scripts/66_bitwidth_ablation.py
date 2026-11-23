# -*- coding: utf-8 -*-
"""
sim 码宽消融：20 / 32 / 52 位，其余配置完全固定。

为什么必须重跑而不能直接引旧仓库
--------------------------------
`Sauomore/Cabinet` 里有 HSH-20（feat4+sim8+abs8）与 HSH-32（feat4+sim20+abs8）
的实测，但那批数字同时变了四个变量：位宽、编码方式（K-means → PCA+sign → DeepHash）、
有无后处理、指标口径（Recall@20 暴力真值 vs overlap@10 教师真值）。
因此它只能作为**演进叙事**引用，不能作为**受控消融**。

本实验固定：教师真值、学生输入（同一 512 维缓存）、隐藏维度、种子、epochs、
后处理超参（pos_k=10, neg_weight=1.0, max_iters=10），只改 n_bits。

2×3 设计：{PCA+sign, DeepHash} × {20, 32, 52} 位，各测裸码与后处理。
其中 PCA+sign 是关键对照——它回答「位宽本身带来多少收益」，
DeepHash 与它的差回答「学习带来多少收益」。

附带：复现旧仓库的历史数字（bge-small 512 维、PCA+sign、sim=20 位、Recall@20），
用于验证那批数据可信。

用法
----
    python scripts/66_bitwidth_ablation.py
    python scripts/66_bitwidth_ablation.py --tag nat16000
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

try:
    import torch
    from cabinet_bnn.hsh.deep_hash import (  # noqa: E402
        bits_to_u64, project, quantize_bits, train_deep_hash, export_deep_hash,
    )
    from cabinet_bnn.hsh.post_optimize import (  # noqa: E402
        greedy_flip_optimize, hamming_matrix,
    )
except Exception as e:  # pragma: no cover
    print(f"  ❌ 导入失败：{type(e).__name__}: {e}")
    raise SystemExit(1)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED

WIDTHS = [20, 32, 52]
KS = (1, 5, 10, 20)


# ---------------------------------------------------------------- 读取
def read_cache(path: Path) -> tuple[int, list[str], np.ndarray]:
    with path.open("rb") as f:
        magic, _ver, dim, count = struct.unpack(">4I", f.read(16))
        if magic != CACHE_MAGIC:
            raise ValueError(f"{path.name}: magic 0x{magic:X}")
        words, vecs = [], np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            (blen,) = struct.unpack(">H", f.read(2))
            words.append(f.read(blen).decode("utf-8"))
            vecs[i] = np.frombuffer(f.read(dim * 4), dtype=">f4").astype(np.float32)
    return dim, words, vecs


# ---------------------------------------------------------------- 评测
def recall_with_dist(pos: np.ndarray, dist: np.ndarray, k: int) -> float:
    """与 cabinet_bnn.hsh.post_optimize.recall_at_k_from_ranking 逐位一致。"""
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


def eval_bits(bits: np.ndarray, sim: np.ndarray) -> dict:
    """一次 Hamming 矩阵，算全部 K 与两种口径。"""
    d = hamming_matrix(bits)
    out = {}
    for k in KS:
        kk = min(k, sim.shape[0] - 1)
        pos = np.argsort(-sim, axis=1, kind="stable")[:, 1 : kk + 1]
        out[f"overlap@{k}"] = recall_with_dist(pos, d, kk)
    pos10 = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    for k in KS:
        out[f"fixed10@{k}"] = recall_with_dist(pos10, d, k)
    return out


def cosine_sim(x: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return xn @ xn.T


def pca_sign(teacher: np.ndarray, width: int, seed: int = 0) -> np.ndarray:
    """PCA 到 width 维后取符号（论文 PCA-52 基线的推广）。"""
    from sklearn.decomposition import PCA
    p = PCA(n_components=width, random_state=seed)
    z = p.fit_transform(teacher)
    return (z > 0).astype(np.uint8)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nat3000")
    ap.add_argument("--teacher", type=Path, default=None, help="教师缓存（默认 data/eval/teacher_{tag}.cache）")
    ap.add_argument("--student", type=Path, default=None, help="学生缓存（默认 data/eval/student_{tag}_d512.cache）")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=RESULTS / "bitwidth_ablation.json")
    args = ap.parse_args()

    tag = args.tag
    tpath = args.teacher or (EVAL / f"teacher_{tag}.cache")
    spath = args.student or (EVAL / f"student_{tag}_d512.cache")
    if not tpath.exists() or not spath.exists():
        print(f"  ❌ 缺缓存：{tpath} / {spath}")
        return 1

    print("=" * 78)
    print(f"  sim 码宽消融 —— {tag}")
    print("=" * 78)

    dim_t, words, teacher = read_cache(tpath)
    dim_s, swords, student = read_cache(spath)
    assert words == swords, "教师与学生缓存的词序不一致"
    n = teacher.shape[0]
    print(f"  教师 {dim_t}d / 学生 {dim_s}d / {n} 词")
    teacher_sim = cosine_sim(teacher)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备 {dev}")

    out = {"tag": tag, "n": n, "teacher_dim": dim_t, "student_dim": dim_s,
           "widths": WIDTHS, "results": {}}

    for width in WIDTHS:
        r = {}
        print(f"\n  ── sim = {width} 位 ──")

        # ---------- A. PCA + sign（零学习对照）
        t0 = time.time()
        bits_pca = pca_sign(teacher, width)
        print(f"    PCA+sign 编码 {time.time() - t0:.1f}s")
        r["pca_bare"] = eval_bits(bits_pca, teacher_sim)
        print(f"      裸码      overlap@10 = {r['pca_bare']['overlap@10']:.4f}"
              f"   @20 = {r['pca_bare']['overlap@20']:.4f}")
        t0 = time.time()
        bits_pca_opt = greedy_flip_optimize(
            bits_pca, teacher_sim, pos_k=10, k=10,
            neg_weight=1.0, max_iters=10, verbose=False)
        r["pca_post"] = eval_bits(bits_pca_opt, teacher_sim)
        print(f"      后处理    overlap@10 = {r['pca_post']['overlap@10']:.4f}"
              f"   @20 = {r['pca_post']['overlap@20']:.4f}   ({time.time() - t0:.1f}s)")

        # ---------- B. DeepHash
        t0 = time.time()
        mean, model, hist = train_deep_hash(
            student, n_bits=width, hidden_dim=args.hidden_dim,
            epochs=args.epochs, lr=1e-3, pos_k=10,
            teacher_x=teacher, seed=args.seed, device=dev, log_every=0,
            verbose=False)
        print(f"    DeepHash 训练 {time.time() - t0:.1f}s")
        u = project(model, mean, student, device=dev)
        bits_dh = quantize_bits(u)
        uniq = len(np.unique(bits_to_u64(bits_dh)))
        r["deephash_unique"] = uniq
        r["deephash_bare"] = eval_bits(bits_dh, teacher_sim)
        print(f"      裸码      overlap@10 = {r['deephash_bare']['overlap@10']:.4f}"
              f"   唯一码 {uniq}/{n} = {uniq / n:.2%}")
        t0 = time.time()
        bits_dh_opt = greedy_flip_optimize(
            bits_dh, teacher_sim, pos_k=10, k=10,
            neg_weight=1.0, max_iters=10, verbose=False)
        r["deephash_post"] = eval_bits(bits_dh_opt, teacher_sim)
        print(f"      后处理    overlap@10 = {r['deephash_post']['overlap@10']:.4f}"
              f"   @20 = {r['deephash_post']['overlap@20']:.4f}   ({time.time() - t0:.1f}s)")

        out["results"][str(width)] = r

    # ---------- 汇总表
    print("\n" + "=" * 78)
    print("  overlap@10 汇总")
    print("=" * 78)
    print(f"    {'方法':<22}" + "".join(f"{w:>10}" for w in WIDTHS))
    for key, name in (("pca_bare", "PCA+sign 裸码"),
                      ("pca_post", "PCA+sign + 后处理"),
                      ("deephash_bare", "DeepHash 裸码"),
                      ("deephash_post", "DeepHash + 后处理")):
        vals = [out["results"][str(w)][key]["overlap@10"] for w in WIDTHS]
        print(f"    {name:<22}" + "".join(f"{v:>10.4f}" for v in vals))
    print()
    print("  overlap@20 汇总（与旧仓库历史口径对照用）")
    print(f"    {'方法':<22}" + "".join(f"{w:>10}" for w in WIDTHS))
    for key, name in (("pca_bare", "PCA+sign 裸码"),
                      ("pca_post", "PCA+sign + 后处理"),
                      ("deephash_bare", "DeepHash 裸码"),
                      ("deephash_post", "DeepHash + 后处理")):
        vals = [out["results"][str(w)][key]["overlap@20"] for w in WIDTHS]
        print(f"    {name:<22}" + "".join(f"{v:>10.4f}" for v in vals))

    # ---------- 历史数字复现
    hist = None
    hp = EVAL / "student_repro3109_d512.cache"
    if hp.exists():
        print("\n" + "=" * 78)
        print("  复现旧仓库历史数字（bge-small 512d，PCA+sign，sim=20 位，Recall@20，暴力真值）")
        print("=" * 78)
        _, _, s3109 = read_cache(hp)
        sim3109 = cosine_sim(s3109)
        b20 = pca_sign(s3109, 20)
        d = hamming_matrix(b20)
        pos20 = np.argsort(-sim3109, axis=1, kind="stable")[:, 1:21]
        rec20 = recall_with_dist(pos20, d, 20)
        print(f"    纯 HSH-32（PCA，20 位 sim，3,109 词）Recall@20 = {rec20:.4f}")
        print(f"    旧仓库记录                                 = 0.2270")
        print(f"    差                                         = {rec20 - 0.227:+.4f}")
        hist = {"recall20_pca20_sim": rec20, "recorded": 0.227}
        out["historical_replication"] = hist

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
