# -*- coding: utf-8 -*-
"""
检测词表语言构成。

为什么需要：
    混合词表（中英同表）在进入 embedding 环节时，需要知道每个词的语言。
    虽然 bge-m3 是多语言的、可以整表编码，但：
      1. 报告语言构成本身就是必要的质量检查
      2. 后续若要对比「中文专用模型 vs 多语言模型」需要分组
      3. 词性标注也要按语言分派（jieba vs nltk）

用法：
    python scripts/47_detect_lang.py --vocab tests/data/vocab_3109.txt
    python scripts/47_detect_lang.py --vocab data/mixed_vocab.txt --out data/mixed_lang.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: CJK 统一表意文字（基本区 + 扩展 A）
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
#: 拉丁字母
LATIN = re.compile(r"[A-Za-z]")
#: 其他文字系统（用于报出未支持的语种）
OTHER_SCRIPTS = {
    "kana": re.compile(r"[\u3040-\u30ff]"),
    "hangul": re.compile(r"[\uac00-\ud7af]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "greek": re.compile(r"[\u0370-\u03ff]"),
    "thai": re.compile(r"[\u0e00-\u0e7f]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
}


def detect(word: str) -> str:
    """返回 'zh' | 'en' | 'mixed' | 'other' | 'empty'"""
    has_cjk = bool(CJK.search(word))
    has_latin = bool(LATIN.search(word))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    if not word.strip():
        return "empty"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="词表语言构成检测")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="可选：写出 word<TAB>lang")
    args = ap.parse_args()

    words = [l.strip() for l in
             args.vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("=" * 70)
    print("词表语言构成")
    print("=" * 70)
    print(f"  文件 {args.vocab}")
    print(f"  词数 {len(words):,}")

    langs = [detect(w) for w in words]
    cnt = Counter(langs)
    print()
    print(f"  {'语言':<10}{'数量':>8}{'占比':>9}   示例")
    print("  " + "-" * 62)
    for lang, n in cnt.most_common():
        ex = [w for w, l in zip(words, langs) if l == lang][:6]
        print(f"  {lang:<10}{n:>8,}{n/len(words)*100:>8.1f}%   {'、'.join(ex)}")

    # 其他文字系统细分
    other_words = [w for w, l in zip(words, langs) if l == "other"]
    if other_words:
        scripts = Counter()
        for w in other_words:
            for name, pat in OTHER_SCRIPTS.items():
                if pat.search(w):
                    scripts[name] += 1
                    break
            else:
                scripts["未知符号"] += 1
        print()
        print("  other 细分:")
        for name, n in scripts.most_common():
            ex = [w for w in other_words
                  if any(p.search(w) for k, p in OTHER_SCRIPTS.items() if k == name)
                  or name == "未知符号"][:5]
            print(f"    {name:<12}{n:>6}   {'、'.join(ex)}")

    # 可疑项
    empties = [w for w, l in zip(words, langs) if l == "empty"]
    if empties:
        print(f"\n  ⚠️ 空行/空白 {len(empties)} 个（已在读取时过滤，不应出现）")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for w, l in zip(words, langs):
                f.write(f"{w}\t{l}\n")
        print(f"\n  已写出 {args.out}")

    out_json = args.vocab.with_suffix(".langstats.json")
    out_json.write_text(json.dumps({
        "vocab": str(args.vocab), "total": len(words),
        "counts": dict(cnt),
        "mixed_words": [w for w, l in zip(words, langs) if l == "mixed"][:50],
        "other_words": other_words[:50],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  统计 -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
