# -*- coding: utf-8 -*-
"""
查清楚：为什么合成词表的裸码召回（0.499）远高于自然词表（0.246）？

之前的假设（合成词近邻依赖共享前缀）已被证伪 ——
合成 16.3% vs 自然 19.1%，合成反而更低。

现在测三个可能的机制：
  H1 近邻更「紧凑」：top-10 与查询的平均余弦更高
  H2 边界更「清晰」：第 10 近邻与第 11 近邻的余弦差更大
  H3 语义更「冗余」：词表内部的平均两两余弦更高（簇更密）

做法：对两个词表的 teacher 嵌入直接算这些量。无需模型。

用法：
    python scripts/60_why_synthetic_easier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402

EVAL = ROOT / "data/eval"


def probe(vocab: Path, cache: Path, k: int = 10) -> dict | None:
    if not vocab.exists() or not cache.exists():
        return None
    words = [l.strip() for l in vocab.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    dim, tw, teacher = read_cache(cache)
    if tw != words:
        idx = {w: i for i, w in enumerate(tw)}
        teacher = teacher[[idx[w] for w in words]]
    tn = normalize_rows(teacher)
    sim = tn @ tn.T
    n = len(words)
    np.fill_diagonal(sim, -np.inf)

    sorted_sim = np.sort(sim, axis=1)[:, ::-1]      # 每行降序
    top10 = sorted_sim[:, :k].mean()
    tenth = sorted_sim[:, k - 1]
    eleventh = sorted_sim[:, k]
    gap = (tenth - eleventh).mean()

    # 词表整体冗余度：随机抽 2000 对算平均余弦
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, 2000)
    b = rng.integers(0, n, 2000)
    ok = a != b
    pair_cos = np.mean([tn[a[i]] @ tn[b[i]] for i in np.where(ok)[0]])

    return {
        "name": vocab.stem, "n": n, "dim": dim,
        "top10_cos": float(top10),
        "cos_10th": float(tenth.mean()),
        "cos_11th": float(eleventh.mean()),
        "gap_10_11": float(gap),
        "pair_cos": float(pair_cos),
    }


def main() -> int:
    print("=" * 82)
    print("为什么合成词表更容易？")
    print("=" * 82)
    print()

    cases = [
        ("合成 3109", ROOT / "tests/data/vocab_3109.txt",
         Path(r"F:\Cabinet-BNN\results\bge_large_3109.cache")),
        ("自然 3000", ROOT / "data/natural_vocab_3000.txt",
         EVAL / "teacher_nat3000.cache"),
        ("自然 10000", ROOT / "data/natural_vocab_10000.txt",
         EVAL / "teacher_nat10000.cache"),
    ]

    rows = []
    print(f"  {'词表':<12}{'N':>7}{'top10余弦':>11}{'第10近邻':>11}"
          f"{'第11近邻':>11}{'10-11间隙':>11}{'随机对余弦':>11}")
    print("  " + "-" * 73)
    for name, v, c in cases:
        r = probe(v, c)
        if r is None:
            print(f"  {name:<12}  缺文件，跳过")
            continue
        r["name"] = name
        rows.append(r)
        print(f"  {name:<12}{r['n']:>7,}{r['top10_cos']:>11.4f}"
              f"{r['cos_10th']:>11.4f}{r['cos_11th']:>11.4f}"
              f"{r['gap_10_11']:>11.4f}{r['pair_cos']:>11.4f}")

    print()
    print("=" * 82)
    print("解读")
    print("=" * 82)
    syn = next((r for r in rows if "合成" in r["name"]), None)
    nat3 = next((r for r in rows if "自然 3000" in r["name"]), None)
    if syn and nat3:
        print(f"  合成 vs 自然3k 对比：")
        print(f"    随机对余弦    {syn['pair_cos']:.4f}  vs  {nat3['pair_cos']:.4f} "
              f"  (差 {syn['pair_cos']-nat3['pair_cos']:+.4f})")
        print(f"    top10 余弦    {syn['top10_cos']:.4f}  vs  {nat3['top10_cos']:.4f} "
              f"  (差 {syn['top10_cos']-nat3['top10_cos']:+.4f})")
        print(f"    10-11 间隙    {syn['gap_10_11']:.4f}  vs  {nat3['gap_10_11']:.4f} "
              f"  (差 {syn['gap_10_11']-nat3['gap_10_11']:+.4f})")
        print()
        # 相对间隙：间隙 / (1 - 随机基线)，衡量「可分性」
        def rel(r):
            span = max(1e-6, 1.0 - r["pair_cos"])
            return r["top10_cos"], r["gap_10_11"] / span
        st, sg = rel(syn)
        nt, ng = rel(nat3)
        print(f"  归一化间隙（间隙 / 语义跨度）:")
        print(f"    合成 {sg:.4f}   自然 {ng:.4f}   比值 {sg/max(ng,1e-9):.2f}x")
        print()
        if sg > ng * 1.5:
            print("  >> 合成词表的 top-10 边界明显更清晰（归一化间隙更大）")
            print("     这解释了它更高的裸码召回：Ranking 更容易。")
        elif syn["top10_cos"] > nat3["top10_cos"] + 0.05:
            print("  >> 合成词表的近邻本身更相似（top10 余弦更高）")
            print("     同时整体冗余度也更高 —— 语义空间更「窄」。")
        else:
            print("  >> 两个机制都不明显；差异可能来自训练数据与随机性。")
            print("     注意：合成词表只有 ~1,114 个基础词 × 修饰词，")
            print("     语义多样性天然低于自然词表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
