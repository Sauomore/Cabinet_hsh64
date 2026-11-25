# -*- coding: utf-8 -*-
"""
E. 后处理超参敏感性

问题
----
当前四个编码器共用同一组后处理超参（pos_k=10, neg_weight=1.0, max_iters=10），
而这三个值最初是在 DeepHash 上定的。若每个编码器的最优超参不同，
那么「52 位上 ITQ(0.7974) > PCA(0.7800)」这个只差 0.0176 的排序，
可能只是超参没调好的假象。

做法
----
52 位，编码器 = PCA+sign / ITQ / DeepHash（seed 42）。
以基线 (10, 1.0, 10) 为起点做坐标扫描（一次只动一个参数）：

    pos_k     ∈ {5, 10, 15, 20}
    neg_weight∈ {0.5, 1, 2, 4}
    max_iters ∈ {5, 10, 20}

再对各编码器取「自身最优」的组合跑一次联合点。
最后回答两个问题：
    ① 每个编码器的最优召回是多少
    ② 用固定超参的排序 vs 各自最优的排序，是否一致

用法
----
    python scripts/76_postproc_sensitivity.py --teacher F:/Cabinet-BNN/results/bge_large_3109.cache
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
BASE = {"pos_k": 10, "neg_weight": 1.0, "max_iters": 10}
GRID = {
    "pos_k": [5, 10, 15, 20],
    "neg_weight": [0.5, 1.0, 2.0, 4.0],
    "max_iters": [5, 10, 20],
}


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


def pca_sign(teacher, r):
    from sklearn.decomposition import PCA
    return (PCA(n_components=r, random_state=0).fit_transform(teacher.astype(np.float64)) >= 0).astype(np.uint8)


def load_itq_module():
    import importlib.util
    src = ROOT / "scripts" / "70_itq_baseline.py"
    spec = importlib.util.spec_from_file_location("itqmod", src)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(src)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--student", type=Path,
                    default=EVAL / "student_repro3109_d512.cache")
    ap.add_argument("--bits", type=int, default=52)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=RESULTS / "postproc_sensitivity.json")
    args = ap.parse_args()

    dim, words, teacher = read_cache(args.teacher)
    n = teacher.shape[0]
    tn = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)
    sim = tn @ tn.T
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    print("=" * 92)
    print(f"  E. 后处理超参敏感性 —— {n} 词 / {args.bits} 位")
    print("=" * 92)

    r = args.bits
    print("  构造三个编码器的码 …")
    codes = {}
    codes["PCA"] = pca_sign(teacher, r)
    itqmod = load_itq_module()
    Y, _ = itqmod.pca_project(teacher, r)
    bits_itq, _R, _d = itqmod.itq(Y, n_iter=50, seed=0)
    codes["ITQ"] = bits_itq
    if not args.student.exists():
        print(f"  ❌ 缺学生缓存 {args.student}")
        return 1
    _, _, student = read_cache(args.student)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _mean, model, _h = train_deep_hash(student, n_bits=r, hidden_dim=512, epochs=500,
                                       lr=1e-3, pos_k=10, teacher_x=teacher,
                                       seed=args.seed, device=dev, log_every=0, verbose=False)
    codes["DeepHash"] = quantize_bits(project(model, _mean, student, device=dev))
    for k, v in codes.items():
        print(f"    {k:<9} 唯一码率 {len(np.unique(v.astype(np.int64) @ (1 << np.arange(r, dtype=np.int64))))/n:.2%}")

    out = {"n": n, "bits": r, "baseline": BASE, "grid": GRID, "results": {}}
    t_all = time.time()

    for name, bits in codes.items():
        print(f"\n  ── {name} ──")
        res = {}

        def run(cfg):
            key = f"pk{cfg['pos_k']}_nw{cfg['neg_weight']}_it{cfg['max_iters']}"
            if key in res:
                return res[key]
            t0 = time.time()
            opt = greedy_flip_optimize(bits, sim, pos_k=cfg["pos_k"], k=10,
                                       neg_weight=cfg["neg_weight"],
                                       max_iters=cfg["max_iters"], verbose=False)
            rec = eval_bits(opt, pos)
            res[key] = {"recall": rec, "cfg": dict(cfg), "seconds": time.time() - t0}
            print(f"    {key:<26} recall@10 {rec:.4f}  ({time.time()-t0:.0f}s)")
            return res[key]

        base_key = f"pk{BASE['pos_k']}_nw{BASE['neg_weight']}_it{BASE['max_iters']}"
        run(BASE)
        # 坐标扫描：一次只动一个参数
        for param, values in GRID.items():
            for v in values:
                if v == BASE[param]:
                    continue
                cfg = dict(BASE)
                cfg[param] = v
                run(cfg)

        best_key = max(res, key=lambda k: res[k]["recall"])
        best = res[best_key]
        # 联合点
        joint = run(best["cfg"]) if best["cfg"] != BASE else base_key
        out["results"][name] = {
            "baseline": res[base_key]["recall"],
            "best": res[best_key]["recall"],
            "best_key": best_key,
            "best_cfg": best["cfg"],
            "gain_over_baseline": res[best_key]["recall"] - res[base_key]["recall"],
            "sweep": res,
        }
        print(f"    → 基线 {res[base_key]['recall']:.4f} / 自身最优 {res[best_key]['recall']:.4f}"
              f"（{best_key}）增益 {res[best_key]['recall'] - res[base_key]['recall']:+.4f}")

    # ---------- 排序对比
    print("\n" + "=" * 92)
    print("  排序是否因调参而改变")
    print("=" * 92)
    print(f"    {'编码器':<10}{'固定超参':>11}{'自身最优':>11}{'增益':>9}   最优配置")
    rank_fixed = sorted(out["results"], key=lambda k: -out["results"][k]["baseline"])
    rank_best = sorted(out["results"], key=lambda k: -out["results"][k]["best"])
    for k in rank_fixed:
        v = out["results"][k]
        print(f"    {k:<10}{v['baseline']:>11.4f}{v['best']:>11.4f}"
              f"{v['gain_over_baseline']:>+9.4f}   {v['best_key']}")
    print(f"\n    固定超参排序： {' > '.join(rank_fixed)}")
    print(f"    各自最优排序： {' > '.join(rank_best)}")
    print(f"    {'✅ 排序一致' if rank_fixed == rank_best else '⚠️ 排序发生变化'}")
    out["rank_fixed"] = rank_fixed
    out["rank_best"] = rank_best
    out["rank_stable"] = bool(rank_fixed == rank_best)

    out["seconds"] = time.time() - t_all
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
