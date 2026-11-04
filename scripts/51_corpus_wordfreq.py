# -*- coding: utf-8 -*-
"""
测 jieba 在真实语料上的吞吐，并抽样看能提取多少高质量词。

为什么需要：
    中文侧要扩词表，最现实的办法是从现有 3.91 GB 语料统计词频。
    但 jieba 是纯 Python 实现，3.91 GB 可能很慢 —— 先测速再定方案。

做法：
    读语料前 N MB，逐行 jieba.cut，统计词频与词性。
    报告：
      · 吞吐（MB/s）
      · 3.91 GB 的预计耗时
      · 不同频次阈值下的候选词量
      · 抽样看看高频词是否合理（防止提取到一堆数字/标点）

用法：
    python scripts/51_corpus_wordfreq.py --sample-mb 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: 只保留纯中文/纯英文的词，滤掉数字、标点、混合串
RE_ZH = re.compile(r"^[\u4e00-\u9fff]{2,6}$")
RE_EN = re.compile(r"^[a-z]{3,20}$")
#: 停用词（高频但无检索价值）
STOPWORDS = set("""
的 了 是 在 和 有 就 都 而 及 与 着 或 一个 我们 你们 他们 它们 这个 那个
这些 那些 什么 怎么 为什么 因为 所以 但是 如果 虽然 可以 没有 不是 就是
自己 时候 问题 方法 情况 方面 进行 通过 以及 并且 或者 而且 然后 这样 那样
一种 一些 一样 这个 那里 这里 现在 已经 还是 只是 不过 可能 应该 需要
""".split())


def main() -> int:
    ap = argparse.ArgumentParser(description="语料词频统计测速")
    ap.add_argument("--corpus", type=Path,
                    default=ROOT.parent / "Cabinet-BNN/data/corpus_all/text_all.txt")
    ap.add_argument("--sample-mb", type=int, default=64,
                    help="抽样读取多少 MB（0 = 全量）")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/corpus_wordfreq_sample.json")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"找不到 {args.corpus}")
        return 1

    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)

    total_bytes = args.corpus.stat().st_size
    limit = args.sample_mb * 1024 * 1024 if args.sample_mb else total_bytes

    print("=" * 74)
    print("语料词频统计（测速 + 抽样）")
    print("=" * 74)
    print(f"  语料 {args.corpus}  ({total_bytes/1024**3:.2f} GB)")
    print(f"  抽样 {limit/1024**2:.0f} MB"
          + ("（全量）" if not args.sample_mb else ""))

    freq: Counter[str] = Counter()
    tag_freq: Counter[str] = Counter()
    tag_of: dict[str, str] = {}
    n_lines = n_chars = 0
    t0 = time.time()

    with args.corpus.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_lines += 1
            n_chars += len(line)
            for w, tag in pseg.cut(line):
                w = w.strip().lower()
                if not w or w in STOPWORDS:
                    continue
                if RE_ZH.match(w) or RE_EN.match(w):
                    freq[w] += 1
                    if w not in tag_of:
                        tag_of[w] = tag
            if n_chars >= limit:
                break

    dt = time.time() - t0
    mb = n_chars / 1024**2
    print(f"\n  完成: {n_lines:,} 行  {mb:.1f} MB  {dt:.1f}s")
    print(f"  吞吐: {mb/dt:.2f} MB/s")
    est_full = total_bytes / 1024**2 / (mb / dt) / 60
    print(f"  全量 {total_bytes/1024**3:.2f} GB 预计: {est_full:.0f} 分钟")

    print(f"\n  去重词数: {len(freq):,}")
    for t, c in Counter(tag_of.values()).most_common(10):
        print(f"    {t:<6} {c:>7,}")

    print("\n  不同频次阈值的候选量（这是词表规模的实际上限）:")
    for th in (2, 3, 5, 10, 20, 50, 100):
        n = sum(1 for c in freq.values() if c >= th)
        print(f"    频次 >= {th:<4}  {n:>8,} 词")

    print("\n  高频词抽样（前 40，人工检查是否合理）:")
    top = freq.most_common(40)
    for i in range(0, len(top), 8):
        chunk = "  ".join(f"{w}({c})" for w, c in top[i:i+8])
        print(f"    {chunk}")

    # 估算：需要多少语料才能得到 N 个词
    print("\n  外推到全量语料:")
    scale = total_bytes / max(n_chars, 1)
    for th in (5, 10, 20):
        n_sample = sum(1 for c in freq.values() if c >= th)
        # 幂律下词数不按比例增长，只给粗略下界
        print(f"    频次 >= {th}: 抽样 {n_sample:,} 词 × 语料倍数 {scale:.0f}"
              f" → 全量至少 {n_sample:,} 词（实际更多）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "corpus": str(args.corpus),
        "sample_mb": mb, "seconds": dt, "throughput_mbs": mb / dt,
        "est_full_minutes": est_full,
        "n_lines": n_lines, "n_unique": len(freq),
        "tag_distribution": dict(Counter(tag_of.values())),
        "by_threshold": {str(th): sum(1 for c in freq.values() if c >= th)
                         for th in (2, 3, 5, 10, 20, 50, 100)},
        "top300": [{"w": w, "c": c, "tag": tag_of.get(w, "")}
                   for w, c in freq.most_common(300)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
