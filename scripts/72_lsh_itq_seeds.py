# -*- coding: utf-8 -*-
"""
补两块对照：LSH（地板基线）与 ITQ 的多随机种子方差。

为什么补这两块
--------------
1. **LSH 是哈希论文里最常被要求的基线**，此前一次都没测。它是数据无关的随机超平面投影，
   预期显著低于任何使用语义结构的编码器；把它放进去才能说明「学/投影语义结构有用」。
2. **ITQ 的旋转是随机初始化的**，此前只跑了 seed 0 的 0.7964。若不报方差，
   「ITQ 在 52 位最好」这个结论站不住。本脚本用与 DeepHash 相同的 3 个种子。

统一设置（与 scripts/67_bitwidth_sweep.py 完全一致）
--------------------------------------------------
真值 = 教师余弦 top-10（排除自身）；重叠口径 overlap@10；稳定排序；
后处理 = greedy_flip_optimize(pos_k=10, k=10, neg_weight=1.0, max_iters=10)。

用法
----
    python scripts/72_lsh_itq_seeds.py --teacher F:/Cabinet-BNN/results/bge_large_3109.cache
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
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
DEFAULT_WIDTHS = [8, 12, 16, 20, 32, 40, 48, 52]
DEFAULT_SEEDS = [42, 2024, 2025]


def _load_itq_module():
    src = ROOT / "scripts" / "70_itq_baseline.py"
    spec = importlib.util.spec_from_file_location("itqmod", src)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(src)
    spec.loader.exec_module(mod)
    return mod


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


def lsh_bits(teacher: np.ndarray, r: int, seed: int) -> np.ndarray:
    """随机超平面 LSH：w ~ N(0, I)，b = sign(x·w)。数据无关。"""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((teacher.shape[1], r)).astype(np.float32)
    return (teacher @ W >= 0).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--tag", default="synth3109")
    ap.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", type=Path, default=RESULTS / "lsh_itq_seeds.json")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    itqmod = _load_itq_module()
    dim, words, teacher = itqmod.read_cache(args.teacher)
    n = teacher.shape[0]
    sim = itqmod.cosine_sim(teacher)
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    print("=" * 88)
    print(f"  LSH / ITQ 多种子对照 —— {args.tag}   教师 {dim}d / {n} 词")
    print(f"  位宽 {widths}   种子 {seeds}")
    print("=" * 88)

    out = {"tag": args.tag, "n": n, "widths": widths, "seeds": seeds, "results": {}}
    t_all = time.time()

    for r in widths:
        t0 = time.time()
        rec = {}
        # ---------- LSH（数据无关）
        bare, post = [], []
        for sd in seeds:
            b = lsh_bits(teacher, r, sd)
            bare.append(eval_bits(b, pos))
            post.append(eval_bits(
                greedy_flip_optimize(b, sim, pos_k=10, k=10, neg_weight=1.0,
                                     max_iters=10, verbose=False), pos))
        rec["lsh_bare"] = {"mean": float(np.mean(bare)), "std": float(np.std(bare))}
        rec["lsh_post"] = {"mean": float(np.mean(post)), "std": float(np.std(post))}

        # ---------- ITQ（PCA + 随机正交旋转）
        Y, _ = itqmod.pca_project(teacher, r)
        bare_i, post_i = [], []
        for sd in seeds:
            b, _R, d = itqmod.itq(Y, n_iter=args.iters, seed=sd)
            bare_i.append(eval_bits(b, pos))
            post_i.append(eval_bits(
                greedy_flip_optimize(b, sim, pos_k=10, k=10, neg_weight=1.0,
                                     max_iters=10, verbose=False), pos))
        rec["itq_bare"] = {"mean": float(np.mean(bare_i)), "std": float(np.std(bare_i))}
        rec["itq_post"] = {"mean": float(np.mean(post_i)), "std": float(np.std(post_i))}
        rec["itq_post_runs"] = post_i

        out["results"][str(r)] = rec
        print(f"\n  {r:>3} 位  ({time.time() - t0:.0f}s)")
        print(f"    LSH  裸码 {rec['lsh_bare']['mean']:.4f}±{rec['lsh_bare']['std']:.4f}"
              f"   后处理 {rec['lsh_post']['mean']:.4f}±{rec['lsh_post']['std']:.4f}")
        print(f"    ITQ  裸码 {rec['itq_bare']['mean']:.4f}±{rec['itq_bare']['std']:.4f}"
              f"   后处理 {rec['itq_post']['mean']:.4f}±{rec['itq_post']['std']:.4f}")

    # ---------- 与既有数据合并成完整编码器表
    sweep = RESULTS / "bitwidth_sweep.json"
    if sweep.exists():
        sw = json.load(io.open(sweep, encoding="utf-8"))
        if sw.get("n") == n:
            print("\n" + "=" * 88)
            print("  四编码器完整对比（overlap@10，后处理后；LSH/ITQ 为 3 种子均值）")
            print("=" * 88)
            print(f"    {'位宽':<6}{'LSH':>10}{'PCA':>10}{'ITQ':>10}{'DeepHash':>12}{'最优':>10}")
            for r in widths:
                key = str(r)
                if key not in sw["results"]:
                    continue
                rows = {
                    "LSH": out["results"][key]["lsh_post"]["mean"],
                    "PCA": sw["results"][key]["pca_post"],
                    "ITQ": out["results"][key]["itq_post"]["mean"],
                    "DH": sw["results"][key]["deephash_post"]["mean"],
                }
                best = max(rows, key=rows.get)
                print(f"    {r:<6}{rows['LSH']:>10.4f}{rows['PCA']:>10.4f}"
                      f"{rows['ITQ']:>10.4f}{rows['DH']:>12.4f}{best:>10}")
                out.setdefault("comparison", {})[key] = rows

    out["seconds"] = time.time() - t_all
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
