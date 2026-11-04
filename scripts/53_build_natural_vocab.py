# -*- coding: utf-8 -*-
"""
构建【自然】中文词表（用于回应「合成词表难以服众」的批评）。

背景：
    论文的 3,109 词表实测是「10 个修饰词 × 约 200 个基础词」的合成组合
    （大U盘 / 大一月 / 大三月 / 快丝瓜 …），65% 是人造词。
    本脚本用 wordfreq 的官方中文词频表构建真实词表。

关键设计：**嵌套词表**
    3,000 ⊂ 6,000 ⊂ 10,000 ⊂ 16,000
    小词表是大词表的前缀，因此同一批词在不同规模下完全一致 ——
    规模曲线才干净（否则每档词表不同，差异分不清是规模还是词变了）。

过滤规则：
    · 长度 >= 2（单字基本是功能词，不是有意义的检索目标）
    · 全是 CJK 字符（去掉混合串、拉丁、数字）
    · 去停用词（的/是/在/了…）—— 它们高频但不是检索目标
    · 保留词频顺序（wordfreq 已按频次排序）

输出：
    data/natural_vocab_3000.txt / _6000 / _10000 / _16000
    data/natural_vocab.meta.json

用法：
    python scripts/53_build_natural_vocab.py
    python scripts/53_build_natural_vocab.py --sizes 3000 6000 10000 16000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: 纯 CJK（不含扩展区，避免生僻字无 embedding）
RE_CJK = re.compile(r"^[\u4e00-\u9fff]+$")

#: 高频功能词 —— 它们是语法词，不是有意义的检索对象
#:
#: 判据：词频极高但语义极泛（「的」「是」「然后」）。
#: 把它们放进检索词表会让 top-10 邻居全是功能词，测不出语义。
STOPWORDS = set("""
的 了 是 在 有 和 就 都 而 及 与 着 或 被 把 让 从 到 对 为 以 于 因 由
一个 我们 你们 他们 它们 这个 那个 这些 那些 这种 那种 这样 那样
什么 怎么 为什么 因为 所以 但是 如果 虽然 然而 不过 只是 还是 就是
可以 没有 不是 已经 现在 时候 问题 方法 情况 方面 进行 通过 以及
并且 或者 而且 然后 因此 由于 关于 对于 根据 按照 通过 随着
自己 大家 别人 东西 事情 地方 时间 一些 一样 一种 一定 一直 一起
可能 应该 需要 必须 能够 不会 不能 不要 不用 不同 不少 不断
这样 那样 怎样 如何 多少 几个 每个 各种 所有 任何 某些
""".split())


def main() -> int:
    ap = argparse.ArgumentParser(description="构建自然中文词表")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[3000, 6000, 10000, 16000])
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=6)
    ap.add_argument("--pool", type=int, default=120000,
                    help="从 wordfreq 取多少候选来过滤")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    try:
        from wordfreq import top_n_list, zipf_frequency
    except ImportError:
        print("需要 wordfreq: pip install wordfreq")
        return 1

    print("=" * 74)
    print("构建自然中文词表")
    print("=" * 74)

    pool = top_n_list(args.lang, args.pool)
    print(f"  wordfreq 候选 {len(pool):,} 词（{args.lang}）")

    # 过滤
    kept: list[str] = []
    seen: set[str] = set()
    stats = {"too_short": 0, "non_cjk": 0, "stopword": 0, "dup": 0,
             "too_long": 0}
    for w in pool:
        w = w.strip()
        if not w:
            continue
        if len(w) < args.min_len:
            stats["too_short"] += 1
            continue
        if len(w) > args.max_len:
            stats["too_long"] += 1
            continue
        if not RE_CJK.match(w):
            stats["non_cjk"] += 1
            continue
        if w in STOPWORDS:
            stats["stopword"] += 1
            continue
        if w in seen:
            stats["dup"] += 1
            continue
        seen.add(w)
        kept.append(w)

    print(f"  过滤后 {len(kept):,} 词")
    for k, v in stats.items():
        print(f"    {k:<12} {v:>8,}")

    max_size = max(args.sizes)
    if len(kept) < max_size:
        print(f"\n  ⚠️ 可用词数 {len(kept):,} < 目标最大 {max_size:,}")
        print(f"     请加大 --pool（当前 {args.pool:,}）")
        args.sizes = [s for s in args.sizes if s <= len(kept)]
        print(f"     本次只构建: {args.sizes}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"lang": args.lang, "filter": stats, "pool": len(pool),
            "kept": len(kept), "sizes": {}, "nested": True}
    for size in args.sizes:
        words = kept[:size]          # 前缀 → 嵌套
        p = args.out_dir / f"natural_vocab_{size}.txt"
        p.write_text("\n".join(words) + "\n", encoding="utf-8")
        meta["sizes"][str(size)] = {
            "path": str(p), "n": len(words),
            "mean_len": sum(len(w) for w in words) / len(words),
            "single_char": sum(1 for w in words if len(w) == 1),
        }
        print(f"\n  [{size:>6,}] {p.name}")
        print(f"    平均长度 {sum(len(w) for w in words)/len(words):.2f} 字")
        print(f"    前 20: {' '.join(words[:20])}")
        print(f"    第 {size//2}-{size//2+10}: "
              f"{' '.join(words[size//2:size//2+11])}")
        print(f"    末 10: {' '.join(words[-10:])}")

    mp = args.out_dir / "natural_vocab.meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  元数据 -> {mp}")

    # 附带：与合成词表的重叠度（说明两者是不同任务）
    syn = ROOT / "tests/data/vocab_3109.txt"
    if syn.exists():
        s = set(l.strip() for l in syn.read_text(encoding="utf-8").splitlines()
                if l.strip())
        nat = set(kept[:3109])
        inter = s & nat
        print(f"\n  与合成词表 vocab_3109 的重叠: {len(inter)} / 3109 "
              f"= {len(inter)/3109:.1%}")
        print(f"    重叠示例: {' '.join(sorted(inter)[:15])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
