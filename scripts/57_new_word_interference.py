# -*- coding: utf-8 -*-
"""
量化「新加入的词是否真的成为邻居」——判断规模曲线是否可信。

要回答的疑问：
    规模曲线显示 overlap@10 从 0.608（3k）到 0.622（10k）几乎不降。
    有两种可能：
      (a) 码空间确实够用，规模不损害召回  —— 好结论
      (b) 新增的是低频外围词，teacher 的 top-10 里根本没有它们，
          所以「检索更难」这件事从未发生 —— 曲线没有测到应有的难度

    本脚本直接测：对前 3,000 个查询词，看 teacher 的 top-10（在全部 N 词上）
    有多少来自【新加入的】那部分。

    若新词几乎不出现：
        → 说明按词频扩词表只是「加外围词」，规模曲线的解释要打折扣
    若新词占相当比例：
        → 说明任务真的变难了，而码仍然扛住 —— (a) 成立

用法：
    python scripts/57_new_word_interference.py --sizes 6000 10000
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402

EVAL = ROOT / "data/eval"
BASE = 3000          # 基准词表大小（查询集 = 前 BASE 词）


def main() -> int:
    ap = argparse.ArgumentParser(description="新词干扰度量化")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6000, 10000, 16000])
    ap.add_argument("--base", type=int, default=BASE)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/new_word_interference.json")
    args = ap.parse_args()

    print("=" * 78)
    print("新词干扰度：teacher 的 top-10 里有多少是「新加入的词」？")
    print("=" * 78)
    print(f"  查询集 = 前 {args.base:,} 词   k={args.k}")
    print()
    print(f"  {'N':>7}{'新词数':>9}{'top10中新词占比':>17}"
          f"{'含新词的查询占比':>19}{'前10余弦均值':>14}")
    print("  " + "-" * 68)

    rows = []
    for size in args.sizes:
        tc = EVAL / f"teacher_nat{size}.cache"
        if not tc.exists():
            print(f"  {size:>7}  缺少 {tc.name}，跳过")
            continue
        dim, words, teacher = read_cache(tc)
        n = len(words)
        if n != size:
            print(f"  {size:>7}  缓存词数 {n} 不符，跳过")
            continue
        tn = normalize_rows(teacher)
        # 只算查询行，省内存
        sim = tn[:args.base] @ tn.T          # (BASE, N)
        # 排除自身
        for i in range(min(args.base, n)):
            sim[i, i] = -np.inf
        top = np.argsort(-sim, axis=1)[:, :args.k]
        new_frac = (top >= args.base).mean()
        has_new = (top >= args.base).any(axis=1).mean()
        top_cos = np.take_along_axis(sim, top, axis=1).mean()

        rows.append({"N": n, "n_new": n - args.base,
                     "new_in_top10": float(new_frac),
                     "queries_with_new": float(has_new),
                     "top10_cos_mean": float(top_cos)})
        print(f"  {n:>7,}{n-args.base:>9,}{new_frac:>16.1%}"
              f"{has_new:>18.1%}{top_cos:>14.4f}")

    print()
    print("=" * 78)
    print("解读")
    print("=" * 78)
    if not rows:
        print("  无数据")
        return 1

    for r in rows:
        verdict = ("新词是真实邻居，任务确实变难" if r["new_in_top10"] > 0.30
                   else ("新词有一定参与" if r["new_in_top10"] > 0.10
                         else "新词基本不参与近邻，属于外围词"))
        print(f"  N={r['N']:,}: top-10 中新词占 {r['new_in_top10']:.1%} → {verdict}")

    print()
    print("  含义：")
    if all(r["new_in_top10"] < 0.10 for r in rows):
        print("    ⚠️ 按词频扩词表主要是在加「外围词」。")
        print("       因此规模曲线的「召回不降」不能完全解释为『码空间够用』——")
        print("       它同时也说明『这些新词本来就不构成竞争』。")
        print("       要真正压测码空间，应该随机抽词，或专门构造易混词对。")
    else:
        print("    ✅ 新词确实进入了近邻集合，规模曲线是有意义的难度提升。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
