# -*- coding: utf-8 -*-
"""
补齐规模曲线上的 PCA 基线空缺（6k / 10k），并给出四个规模的 PCA+后处理。

背景
----
docs 与论文表 5 的 PCA-52 一列只测了 3k 与 16k，6k/10k 标为「未测」。
更关键的是：该列是 **PCA 裸码**（3k 0.3757、16k 0.3126），
而 HSH 一列是**后处理之后**。两者口径不同，容易被误读为「HSH 大幅领先」。
本脚本对四个规模同时给出 PCA 裸码与 PCA+后处理，使该表同口径可比。

用法
----
    python scripts/73_scale_pca_fill.py
"""

from __future__ import annotations

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

EVAL = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED
TAGS = ["nat3000", "nat6000", "nat10000", "nat16000"]


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
    total = 0.0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        total += np.intersect1d(pos[i], order[:kk], assume_unique=False).size / pos.shape[1]
    return total / n


def eval_bits(bits: np.ndarray, pos: np.ndarray, k: int = 10) -> float:
    return recall_with_dist(pos, hamming_matrix(bits), k)


def main() -> int:
    print("=" * 84)
    print("  规模曲线补齐：PCA-52 的裸码与后处理（四个规模）")
    print("=" * 84)
    out = {"width": 52, "datasets": {}}
    t_all = time.time()

    for tag in TAGS:
        tp = EVAL / f"teacher_{tag}.cache"
        if not tp.exists():
            print(f"\n  [{tag}] 跳过：缺 {tp.name}")
            continue
        t0 = time.time()
        dim, words, teacher = read_cache(tp)
        n = teacher.shape[0]
        tn = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)
        sim = tn @ tn.T
        pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]

        from sklearn.decomposition import PCA
        Z = PCA(n_components=52, random_state=0).fit_transform(teacher.astype(np.float64))
        bits = (Z >= 0).astype(np.uint8)
        rec_bare = eval_bits(bits, pos)
        t_post = time.time()
        opt = greedy_flip_optimize(bits, sim, pos_k=10, k=10, neg_weight=1.0,
                                   max_iters=10, verbose=False)
        rec_post = eval_bits(opt, pos)
        dt = time.time() - t0

        out["datasets"][tag] = {
            "n": n, "pca_bare": rec_bare, "pca_post": rec_post,
            "post_seconds": time.time() - t_post, "seconds": dt,
        }
        print(f"\n  [{tag}]  n={n}")
        print(f"    PCA-52 裸码      {rec_bare:.4f}")
        print(f"    PCA-52 + 后处理  {rec_post:.4f}   （后处理 {time.time() - t_post:.0f}s）")

    # ---------- 与 HSH（DeepHash+后处理）记录值合并
    print("\n" + "=" * 84)
    print("  同口径规模对照（overlap@10）")
    print("=" * 84)
    print(f"    {'词表':<10}{'规模':>8}{'PCA 裸码':>11}{'PCA+后处理':>13}"
          f"{'HSH 裸码':>11}{'HSH+后处理':>13}")
    rows = []
    for tag in TAGS:
        j = EVAL / f"eval_{tag}.json"
        if tag not in out["datasets"] or not j.exists():
            continue
        d = json.load(j.open(encoding="utf-8"))
        rp = d["recall_pre"]["overlap"]["10"]
        ro = d["recall_post"]["overlap"]["10"]
        r = out["datasets"][tag]
        rows.append({"tag": tag, "n": r["n"], "pca_bare": r["pca_bare"],
                     "pca_post": r["pca_post"], "hsh_bare": rp, "hsh_post": ro})
        print(f"    {tag:<10}{r['n']:>8}{r['pca_bare']:>11.4f}{r['pca_post']:>13.4f}"
              f"{rp:>11.4f}{ro:>13.4f}")
    if rows:
        out["comparison"] = rows
        d3, d16 = rows[0], rows[-1]
        print()
        print(f"    3k→16k 变化：  PCA 裸码 {d16['pca_bare'] - d3['pca_bare']:+.4f}"
              f"   PCA+后处理 {d16['pca_post'] - d3['pca_post']:+.4f}"
              f"   HSH+后处理 {d16['hsh_post'] - d3['hsh_post']:+.4f}")

    out["seconds"] = time.time() - t_all
    (RESULTS / "scale_pca_fill.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 results/scale_pca_fill.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
