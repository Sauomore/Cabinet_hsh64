# -*- coding: utf-8 -*-
"""
ITQ 的机制诊断：为何量化误差更低，裸码 Recall@10 反而更低？

现象（见 results/itq_baseline.json）
------------------------------------
52 位下 ITQ 的量化误差比 R=I（即 PCA+sign）低 2.4%，但裸码 Recall@10
是 0.3482，而 PCA+sign 是 0.6125。旋转是等距变换、量化误差更低，
Hamming 本应更准，因此这一反直觉结果需要一个机制解释。

本脚本给出解释，测四件事
------------------------
  ① 同一 r 维 PCA 子空间内「精确 L2」的 Recall@10 上限（旋转不变，两者共享）
  ② 每种码的 Hamming 排序对该 L2 几何的逼近程度：corr(Hamming, L2)
  ③ 每种码的 Hamming 排序对真值（1024 维余弦）的逼近程度：corr(Hamming, −cos)
  ④ 量化误差、唯一码率、比特平衡、旋转幅度

判据
----
若 PCA+sign 的 Hamming 召回**高于**精确 L2 的上限（①），说明符号码并不是
L2 的有损近似，而是一种**方差均衡的角度度量**；真值是余弦（角度），
故角度度量优于幅值敏感的 L2。ITQ 最小化 ||B − YR||² 是幅值敏感的，
把码做得更像 L2 的忠实近似（corr 更高），反而偏离真值。

用法
----
    python scripts/71_itq_diagnose.py --teacher F:/Cabinet-BNN/results/bge_large_3109.cache
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

from cabinet_bnn.hsh.post_optimize import hamming_matrix  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS = ROOT / "results"


def _load_itq_module():
    """把 70_itq_baseline.py 当模块加载（复用其 read_cache / pca_project / itq）。"""
    src = ROOT / "scripts" / "70_itq_baseline.py"
    spec = importlib.util.spec_from_file_location("itqmod", src)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(src)
    spec.loader.exec_module(mod)
    return mod


def recall_at_k(pos: np.ndarray, dist: np.ndarray, k: int) -> float:
    """stable argsort + 排除自身，与项目其余评测逐位一致。"""
    n = dist.shape[0]
    kk = min(k, n - 1)
    total = 0.0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        total += np.intersect1d(pos[i], order[:kk], assume_unique=False).size / pos.shape[1]
    return total / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--widths", default="20,52")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS / "itq_diagnose.json")
    args = ap.parse_args()

    itqmod = _load_itq_module()
    read_cache = itqmod.read_cache
    pca_project = itqmod.pca_project
    itq = itqmod.itq
    cosine_sim = itqmod.cosine_sim

    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    dim, words, teacher = read_cache(args.teacher)
    n = teacher.shape[0]
    sim = cosine_sim(teacher)
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    print("=" * 80)
    print(f"  ITQ 机制诊断  教师 {dim}d / {n} 词")
    print("=" * 80)

    out = {"n": n, "teacher_dim": dim, "results": {}}
    from scipy.stats import spearmanr

    for r in widths:
        Y, _ = pca_project(teacher, r)
        bits_p = (Y >= 0).astype(np.uint8)
        bits_i, R, diag = itq(Y, n_iter=args.iters, seed=args.seed)

        # ① 精确 L2（旋转不变）
        sq = (Y * Y).sum(1)
        d2 = sq[:, None] - 2.0 * (Y @ Y.T) + sq[None, :]
        rec_l2 = recall_at_k(pos, d2, 10)

        hp = hamming_matrix(bits_p)
        hi = hamming_matrix(bits_i)
        rec_hp = recall_at_k(pos, hp, 10)
        rec_hi = recall_at_k(pos, hi, 10)

        # ②③ 两种相关性（抽样 200 个查询）
        rng = np.random.default_rng(0)
        q = rng.choice(n, 200, replace=False)
        cl_p, cl_i, cc_p, cc_i = [], [], [], []
        for qi in q:
            mask = np.ones(n, bool)
            mask[qi] = False
            cl_p.append(spearmanr(hp[qi][mask], d2[qi][mask]).statistic)
            cl_i.append(spearmanr(hi[qi][mask], d2[qi][mask]).statistic)
            cc_p.append(spearmanr(hp[qi][mask], -sim[qi][mask]).statistic)
            cc_i.append(spearmanr(hi[qi][mask], -sim[qi][mask]).statistic)

        rec = {
            "exact_l2_ceiling": rec_l2,
            "pca_hamming": rec_hp,
            "itq_hamming": rec_hi,
            "pca_beats_l2_ceiling": bool(rec_hp > rec_l2),
            "corr_hamming_l2": {"pca": float(np.mean(cl_p)), "itq": float(np.mean(cl_i))},
            "corr_hamming_cosine": {"pca": float(np.mean(cc_p)), "itq": float(np.mean(cc_i))},
            "quant_err": {"itq": diag["err_final"], "identity": diag["err_identity"],
                          "drop_pct": (1 - diag["err_final"] / diag["err_identity"]) * 100},
            "unique_rate": {"pca": float(
                len(np.unique(bits_p.astype(np.int64) @ (1 << np.arange(r, dtype=np.int64)))) / n),
                "itq": diag["unique_rate"]},
            "rotation_norm": float(np.linalg.norm(R - np.eye(r))),
            "bit_balance": {"pca_mean": float(bits_p.mean()), "itq_mean": diag["bit_balance"],
                            "pca_range": [float(bits_p.mean(0).min()), float(bits_p.mean(0).max())],
                            "itq_range": [diag["bit_min"], diag["bit_max"]]},
        }
        out["results"][str(r)] = rec

        print(f"\n  ── {r} 位 ──")
        print(f"    ① 精确 L2@同一子空间上限        {rec_l2:.4f}")
        print(f"      PCA+sign Hamming            {rec_hp:.4f}   "
              f"{'（高于 L2 上限 → 符号码不是 L2 的有损近似）' if rec_hp > rec_l2 else ''}")
        print(f"      ITQ      Hamming            {rec_hi:.4f}")
        print(f"    ② corr(Hamming, L2)    PCA {np.mean(cl_p):.4f}   ITQ {np.mean(cl_i):.4f}"
              f"   ← ITQ 更贴近 L2（符合其优化目标）")
        print(f"    ③ corr(Hamming, −cos)  PCA {np.mean(cc_p):.4f}   ITQ {np.mean(cc_i):.4f}")
        print(f"    量化误差降幅 {rec['quant_err']['drop_pct']:.2f}%"
              f"   ‖R−I‖ {rec['rotation_norm']:.2f}"
              f"   唯一码率 PCA {rec['unique_rate']['pca']:.2%} / ITQ {rec['unique_rate']['itq']:.2%}")
        print(f"    比特平衡均值 PCA {bits_p.mean():.3f} / ITQ {diag['bit_balance']:.3f}")

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
