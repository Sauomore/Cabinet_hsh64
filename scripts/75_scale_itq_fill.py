# -*- coding: utf-8 -*-
"""
补齐 ITQ 在自然词表四个规模上的裸码与后处理。

为什么需要
----------
合成 3,109 基准上，后处理后的最优编码器是 ITQ（0.7974），高于 PCA（0.7800）
与 DeepHash（0.7420）。但规模曲线（scripts/73）目前只算了 PCA。
若自然词表上 ITQ 同样优于 PCA，则论文的规模表应换成「最优编码器 + 后处理」，
而不是「PCA + 后处理」。

设置：与 scripts/73 完全一致 —— 教师嵌入 → PCA-52 → sign（ITQ 再加正交旋转），
后处理 greedy_flip(pos_k=10, k=10, neg_weight=1.0, max_iters=10)，指标 overlap@10。
ITQ 的旋转随机初始化，在 52 位下已验证方差极小（合成基准 ±0.0014），故取单种子。

用法
----
    python scripts/75_scale_itq_fill.py
"""

from __future__ import annotations

import importlib.util
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

EVAL = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED
TAGS = ["nat3000", "nat6000", "nat10000", "nat16000"]


def _load_itq_module():
    src = ROOT / "scripts" / "70_itq_baseline.py"
    spec = importlib.util.spec_from_file_location("itqmod", src)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(src)
    spec.loader.exec_module(mod)
    return mod


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
    itqmod = _load_itq_module()
    print("=" * 88)
    print("  规模曲线补齐：ITQ-52 的裸码与后处理（四个规模）")
    print("=" * 88)
    out = {"width": 52, "method": "itq", "datasets": {}}
    t_all = time.time()

    for tag in TAGS:
        tp = EVAL / f"teacher_{tag}.cache"
        if not tp.exists():
            print(f"\n  [{tag}] 跳过：缺 {tp.name}")
            continue
        t0 = time.time()
        dim, words, teacher = read_cache(tp)
        n = teacher.shape[0]
        sim = itqmod.cosine_sim(teacher)
        pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]

        Y, _ = itqmod.pca_project(teacher, 52)
        bits, R, diag = itqmod.itq(Y, n_iter=50, seed=0)
        bare = eval_bits(bits, pos)
        t_post = time.time()
        opt = greedy_flip_optimize(bits, sim, pos_k=10, k=10, neg_weight=1.0,
                                  max_iters=10, verbose=False)
        post = eval_bits(opt, pos)
        dt = time.time() - t0

        out["datasets"][tag] = {
            "n": n, "itq_bare": bare, "itq_post": post,
            "quant_err_drop_pct": (1 - diag["err_final"] / diag["err_identity"]) * 100,
            "rotation_norm": float(np.linalg.norm(R - np.eye(52))),
            "post_seconds": time.time() - t_post, "seconds": dt,
        }
        print(f"\n  [{tag}]  n={n}")
        print(f"    ITQ-52 裸码      {bare:.4f}")
        print(f"    ITQ-52 + 后处理  {post:.4f}   （后处理 {time.time() - t_post:.0f}s）")
        # 增量落盘：后续若崩溃，已算的规模不丢
        (RESULTS / "scale_itq_fill.partial.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 三路合并：ITQ / PCA / HSH(DeepHash)
    print("\n" + "=" * 92)
    print("  同口径规模对照（overlap@10，全部经后处理）")
    print("=" * 92)
    pca_fill = json.load(io.open(RESULTS / "scale_pca_fill.json", encoding="utf-8")) \
        if (RESULTS / "scale_pca_fill.json").exists() else {"datasets": {}}
    print(f"    {'词表':<10}{'规模':>8}{'HSH(DH)+post':>15}{'PCA+post':>12}"
          f"{'ITQ+post':>12}{'最优':>8}")
    rows = []
    for tag in TAGS:
        if tag not in out["datasets"]:
            continue
        j = EVAL / f"eval_{tag}.json"
        if not j.exists():
            continue
        d = json.load(j.open(encoding="utf-8"))
        hsh = d["recall_post"]["overlap"]["10"]
        pca = pca_fill["datasets"].get(tag, {}).get("pca_post")
        itq = out["datasets"][tag]["itq_post"]
        best = max({"HSH": hsh, "PCA": pca or -1, "ITQ": itq}, key=lambda k: {
            "HSH": hsh, "PCA": pca or -1, "ITQ": itq}[k])
        rows.append({"tag": tag, "n": out["datasets"][tag]["n"], "hsh_post": hsh,
                     "pca_post": pca, "itq_post": itq, "best": best})
        print(f"    {tag:<10}{out['datasets'][tag]['n']:>8}{hsh:>15.4f}"
              f"{(pca if pca is not None else float('nan')):>12.4f}{itq:>12.4f}{best:>8}")
    if rows:
        out["comparison"] = rows
        print()
        for key in ("hsh_post", "pca_post", "itq_post"):
            d3 = rows[0][key]
            d16 = rows[-1][key]
            if d3 is None or d16 is None:
                continue
            print(f"    3k→16k：{key:<10} {d16 - d3:+.4f}")

    out["seconds"] = time.time() - t_all
    (RESULTS / "scale_itq_fill.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 results/scale_itq_fill.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
