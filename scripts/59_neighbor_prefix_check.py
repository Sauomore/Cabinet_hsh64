# -*- coding: utf-8 -*-
"""
验证：合成词表的「真邻居」是否主要由共享前缀决定。

动机：
    自然词表裸码 overlap@10 = 0.246，而合成词表 = 0.499。
    差这么多，怀疑合成词表的近邻结构本身更「简单」：
    大U盘 / 大一月 / 大三月 … 共享「大」前缀，
    教师（bge-large）会把它们排到一起，
    于是「找最近 10 个」几乎等于「找同前缀的词」。

    若成立，则论文的 0.74 是在一个近邻结构被人为简化的任务上取得的。

做法：
    对两个词表各取查询集，算 teacher top-10 中「与查询共享首字」的比例。

用法：
    python scripts/59_neighbor_prefix_check.py
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


def analyse(name: str, vocab: Path, cache: Path, k: int = 10) -> dict:
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
    top = np.argsort(-sim, axis=1)[:, :k]

    # 共享首字
    same_first = 0
    total = 0
    for i in range(n):
        w = words[i]
        for j in top[i]:
            total += 1
            if words[j][0] == w[0]:
                same_first += 1

    # 共享任意长度前缀（>=1）
    # 也统计「完全同前缀 2 字」
    same2 = 0
    for i in range(n):
        w = words[i]
        for j in top[i]:
            if len(w) >= 2 and len(words[j]) >= 2 and words[j][:2] == w[:2]:
                same2 += 1

    return {
        "name": name, "n": n,
        "same_first_char": same_first / total,
        "same_2char_prefix": same2 / total,
        "k": k,
    }


def main() -> int:
    print("=" * 76)
    print("近邻结构检查：合成 vs 自然")
    print("=" * 76)
    print("  问：teacher 的 top-10 里，与查询共享首字/前两字的比例有多高？")
    print("  比例越高 -> 近邻结构越依赖字面前缀，任务越「简单」")
    print()

    cases = [
        ("合成 3,109", ROOT / "tests/data/vocab_3109.txt",
         Path(r"F:\Cabinet-BNN\results\bge_large_3109.cache")),
        ("自然 3,000", ROOT / "data/natural_vocab_3000.txt",
         EVAL / "teacher_nat3000.cache"),
        ("自然 10,000", ROOT / "data/natural_vocab_10000.txt",
         EVAL / "teacher_nat10000.cache"),
    ]

    rows = []
    print(f"  {'词表':<14}{'N':>8}{'共享首字':>12}{'共享前两字':>13}")
    print("  " + "-" * 46)
    for name, vocab, cache in cases:
        if not vocab.exists() or not cache.exists():
            print(f"  {name:<14}  缺少文件，跳过")
            continue
        r = analyse(name, vocab, cache)
        rows.append(r)
        print(f"  {r['name']:<14}{r['n']:>8,}"
              f"{r['same_first_char']:>11.1%}{r['same_2char_prefix']:>12.1%}")

    print()
    if len(rows) >= 2:
        syn = next((r for r in rows if "合成" in r["name"]), None)
        nat = next((r for r in rows if "自然" in r["name"]), None)
        if syn and nat:
            d = syn["same_first_char"] - nat["same_first_char"]
            print(f"  合成词表比自然词表高 {d*100:.1f} 个百分点（共享首字）")
            if syn["same_first_char"] > 0.5 and nat["same_first_char"] < 0.3:
                print("  >> 确认：合成词表的近邻主要由「共享前缀」决定，")
                print("     因此论文的 0.74 是在一个被人为简化的近邻结构上取得的。")
                print("     自然词表上的 0.61 才是更有参考价值的数字。")
            elif d > 0.15:
                print("  >> 合成词表确实更依赖字面前缀，但差距不到决定性程度。")
            else:
                print("  >> 两者近邻结构的字面重叠度差异不大，")
                print("     召回差异应主要来自语义复杂度而非前缀。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
