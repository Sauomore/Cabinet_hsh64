# -*- coding: utf-8 -*-
"""
干扰项实验：把「规模变大」对召回的纯干扰效应分离出来。

为什么需要这个：
    上一步的规模曲线用的是【按词频排序】的嵌套词表。结果召回几乎不随
    规模变化（0.608 -> 0.599 -> 0.594）。这有两种解释：
      (a) HSH-64 的码对规模确实不敏感 —— 很好的结论
      (b) 新加进来的是低频、语义偏远的外围词，本来就不会成为干扰项
          —— 那么曲线测的是「加外围词」，不是「加干扰词」

    本实验区分两者：
      查询集  固定为前 3,000 词
      真值集  固定为 teacher top-10【限定在这 3,000 词内】
      检索集  Hamming top-10【在全部 N 词内】
    于是随着 N 增大，只有干扰项增多，真值定义不变。
    召回下降 = 纯干扰效应。

    对照组：真值集与检索集都用全部 N 词（= 之前的规模曲线口径），
    两相对照就能看出「新词是否真的是干扰项」。

用法：
    python scripts/56_distractor_analysis.py
    python scripts/56_distractor_analysis.py --sizes 3000 6000 10000 16000
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
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from _eval_helpers import load_exported, load_pca_student  # noqa: E402
from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402
from cabinet_bnn.hsh.deep_hash import bits_to_u64, project, quantize_bits  # noqa: E402
from cabinet_bnn.hsh.post_optimize import load_override_cache  # noqa: E402

EVAL = ROOT / "data/eval"


def hamming_to_query(qbits: np.ndarray, dbits: np.ndarray) -> np.ndarray:
    """(Q,52) vs (N,52) -> (Q,N) Hamming 距离，分块。"""
    q, n = qbits.shape[0], dbits.shape[0]
    out = np.empty((q, n), dtype=np.float32)
    blk = max(1, int(8_000_000 // max(n, 1)))
    for s in range(0, q, blk):
        e = min(s + blk, q)
        out[s:e] = (qbits[s:e, None, :] != dbits[None, :, :]).sum(
            axis=2, dtype=np.float32)
    return out


def load_run(size: int, n_query: int):
    """载入某规模的产物，返回 (words, bits, teacher_sim)。"""
    tag = f"nat{size}"
    vocab = ROOT / f"data/natural_vocab_{size}.txt"
    words = [l.strip() for l in vocab.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    # 教师
    tc = EVAL / f"teacher_{tag}.cache"
    if not tc.exists():
        return None
    dim_t, tw, teacher = read_cache(tc)
    if tw != words:
        idx = {w: i for i, w in enumerate(tw)}
        teacher = teacher[[idx[w] for w in words]]
    tn = normalize_rows(teacher)
    ts = tn @ tn.T

    # 学生 + 模型
    student = load_pca_student(teacher, 512, EVAL / f"pca512_{tag}.npz")
    mean, model = load_exported(EVAL / f"deep_hash_{tag}.bin")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    bits = quantize_bits(project(model, mean, student, device=dev))

    # 后处理 override
    ov = EVAL / f"sim_override_{tag}.bin"
    if ov.exists():
        ow, obits = load_override_cache(ov)
        idx = {w: i for i, w in enumerate(ow)}
        opt = np.zeros_like(bits)
        for i, w in enumerate(words):
            if w in idx:
                opt[i] = obits[idx[w]]
        bits_opt = opt
    else:
        bits_opt = bits

    return {"words": words, "raw": bits, "opt": bits_opt,
            "teacher_sim": ts, "n": len(words)}


def eval_variants(run: dict, n_query: int, k: int = 10) -> dict:
    """
    三种口径：
      A 全库口径  : 查询前 n_query 词；真值 teacher top-k（全库）；检索 Hamming top-k（全库）
      B 干扰项口径: 查询前 n_query 词；真值 teacher top-k（限定前 n_query）；
                    检索 Hamming top-k（全库）
      C 库内口径  : 查询前 n_query 词；真值/检索都限定前 n_query
    """
    words = run["words"]
    ts = run["teacher_sim"]
    n = run["n"]
    q = min(n_query, n)
    out = {}

    for name_key in ("raw", "opt"):
        bits = run[name_key]
        # 全库距离：查询前 q 行 vs 全部 n
        d = hamming_to_query(bits[:q], bits)

        # --- A 全库 ---
        truth_full = np.argsort(-ts[:q, :], axis=1)[:, :k + 1]
        recall_a = _recall_by_index(d, truth_full, self_idx=np.arange(q), k=k)

        # --- C 库内 ---
        ts_sub = ts[:q, :q]
        truth_sub = np.argsort(-ts_sub, axis=1)[:, :k + 1]
        recall_c = _recall_by_index(d[:, :q], truth_sub,
                                    self_idx=np.arange(q), k=k)

        # --- B 干扰项（真值限定前 q，检索全体 n）---
        recall_b = _recall_by_index(d, truth_sub, self_idx=np.arange(q), k=k)

        out[name_key] = {"A_full": recall_a, "B_distractor": recall_b,
                         "C_in_subset": recall_c}
    return out


def _recall_by_index(d: np.ndarray, truth_idx: np.ndarray,
                     self_idx: np.ndarray, k: int) -> float:
    """overlap@k：检索集 = Hamming top-k，真值集 = truth_idx 中的前 k 个（去自身）。

    d: (Q, N) 距离矩阵；truth_idx: (Q, k+1) 的 teacher 近邻（首列可能是自身）。
    """
    q = d.shape[0]
    total = 0.0
    for i in range(q):
        me = int(self_idx[i])
        # 检索集
        order = np.argsort(d[i], kind="stable")
        order = order[order != me]
        retrieved = set(int(x) for x in order[:k])
        # 真值集：按 teacher 排序取前 k 个，排除自身
        truth = [int(x) for x in truth_idx[i] if int(x) != me][:k]
        if not truth:
            continue
        total += len(set(truth) & retrieved) / len(truth)
    return total / q


def main() -> int:
    ap = argparse.ArgumentParser(description="干扰项分析")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[3000, 6000, 10000, 16000])
    ap.add_argument("--n-query", type=int, default=3000,
                    help="查询集大小（固定，默认 3000 = 最小词表）")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/distractor_analysis.json")
    args = ap.parse_args()

    print("=" * 78)
    print("干扰项分析：规模变大是「加外围词」还是「加干扰词」？")
    print("=" * 78)
    print(f"  查询集固定前 {args.n_query:,} 词   k={args.k}")
    print()
    print(f"  {'N':>7}{'口径':>26}{'裸码':>10}{'后处理':>10}")
    print("  " + "-" * 56)

    summary = []
    for size in args.sizes:
        run = load_run(size, args.n_query)
        if run is None:
            print(f"  {size:>7}  产物缺失，跳过")
            continue
        r = eval_variants(run, args.n_query, args.k)
        row = {"size": size, **r}
        summary.append(row)
        labels = [("A_full", "A 全库口径"),
                  ("B_distractor", "B 干扰项口径"),
                  ("C_in_subset", "C 库内口径")]
        for i, (key, label) in enumerate(labels):
            n_str = f"{size:,}" if i == 0 else ""
            print(f"  {n_str:>7}{label:>26}"
                  f"{r['raw'][key]:>10.4f}{r['opt'][key]:>10.4f}")

    print()
    print("=" * 78)
    print("解读")
    print("=" * 78)
    if len(summary) >= 2:
        first, last = summary[0], summary[-1]
        for key, label in [("A_full", "A 全库"), ("B_distractor", "B 干扰项"),
                           ("C_in_subset", "C 库内")]:
            d = last["opt"][key] - first["opt"][key]
            print(f"  {label:<12} {first['size']:,} -> {last['size']:,}  "
                  f"后处理召回变化 {d:+.4f}")
        print()
        # B 降而 A 不降 => 新词确实是干扰项，但被训练吸收了
        a_d = last["opt"]["A_full"] - first["opt"]["A_full"]
        b_d = last["opt"]["B_distractor"] - first["opt"]["B_distractor"]
        if abs(a_d) < 0.01 and b_d < -0.01:
            print("  >> 新词【是】干扰项（B 下降），但训练把它们吸收了（A 不变）")
            print("     说明：扩规模需要更多训练数据，而不是码空间不够")
        elif a_d < -0.01 and b_d < -0.01:
            print("  >> 规模确实在损害召回（A、B 都下降）")
        elif abs(a_d) < 0.01 and abs(b_d) < 0.01:
            print("  >> 新词基本不构成干扰（A、B 都平稳）")
            print("     注意：词表按词频排序，新增的是低频外围词，可能天然远离高频词")
        else:
            print(f"  >> 混合效应：A {a_d:+.4f}  B {b_d:+.4f}，需逐档细看")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
