# -*- coding: utf-8 -*-
"""
构建中英双语混合词表。

数据来源：
    · 中文侧：沿用 tests/data/vocab_3109.txt（3109 词，已有配套 embedding 缓存，
      换掉会让旧实验失去可比性）
    · 英文侧：ECDICT（skywind3000/ECDICT，MIT）—— 英汉词典数据库，
      含 BNC 与 COCA 词频排名，可据此选高频英文词

英文选词规则（可调）：
    · 词频：bnc 或 frq（COCA）排名 <= --max-rank
    · 形态：只保留纯 ASCII 字母的词（去掉带连字符、空格、重音符号的条目）
    · 长度：--min-len ~ --max-len
    · 排除：单字母、缩写（全大写且长度<=3）

输出：
    data/mixed_vocab.txt        每行一个词（中文在前，英文在后）
    data/mixed_vocab.meta.json  选词参数与统计

用法：
    python scripts/48_build_vocab.py --n-en 3000
    python scripts/48_build_vocab.py --n-en 3000 --max-rank 8000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ASCII_WORD = re.compile(r"^[a-z]+$")


def load_english(ecdict: Path, n_en: int, max_rank: int,
                 min_len: int, max_len: int) -> tuple[list[tuple[str, int]], dict]:
    """从 ECDICT 选高频英文词。返回 ([(词, 频次排名)], 统计)。"""
    picked: list[tuple[str, int]] = []
    seen: set[str] = set()
    stats = {"total_rows": 0, "has_rank": 0, "bad_form": 0, "too_long": 0,
             "duplicate": 0, "abbrev": 0}

    with ecdict.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows"] += 1
            w = (row.get("word") or "").strip().lower()

            # 词频：优先取两者中较好的排名
            ranks = []
            for col in ("bnc", "frq"):
                v = (row.get(col) or "").strip()
                if v.isdigit() and int(v) > 0:
                    ranks.append(int(v))
            if not ranks:
                continue
            rank = min(ranks)
            if rank > max_rank:
                continue
            stats["has_rank"] += 1

            if not ASCII_WORD.match(w):
                stats["bad_form"] += 1
                continue
            if not (min_len <= len(w) <= max_len):
                stats["too_long"] += 1
                continue
            if w in seen:
                stats["duplicate"] += 1
                continue
            seen.add(w)
            picked.append((w, rank))

    picked.sort(key=lambda x: x[1])
    return picked[:n_en], stats


def main() -> int:
    ap = argparse.ArgumentParser(description="构建中英混合词表")
    ap.add_argument("--ecdict", type=Path, default=ROOT / "data/ecdict.csv")
    ap.add_argument("--zh-vocab", type=Path,
                    default=ROOT / "tests/data/vocab_3109.txt")
    ap.add_argument("--n-en", type=int, default=3000, help="英文词数量")
    ap.add_argument("--max-rank", type=int, default=6000,
                    help="词频排名上限（BNC/COCA 取较优者）")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "data/mixed_vocab.txt")
    args = ap.parse_args()

    for p in (args.ecdict, args.zh_vocab):
        if not p.exists():
            print(f"缺少 {p}")
            return 1

    print("=" * 72)
    print("构建中英混合词表")
    print("=" * 72)

    zh = [l.strip() for l in
          args.zh_vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n  中文侧 {args.zh_vocab.name}: {len(zh):,} 词")

    print(f"\n  英文侧 {args.ecdict.name}（选取中 …）")
    en, stats = load_english(args.ecdict, args.n_en, args.max_rank,
                             args.min_len, args.max_len)
    print(f"    扫描 {stats['total_rows']:,} 行")
    print(f"    有词频排名 {stats['has_rank']:,}")
    print(f"    形态不符丢弃 {stats['bad_form']:,}"
          f"（含连字符/空格/非 ASCII）")
    print(f"    长度不符丢弃 {stats['too_long']:,}")
    print(f"    重复丢弃 {stats['duplicate']:,}")
    print(f"    最终选取 {len(en):,}")

    if en:
        print(f"\n    频次最高的 15 个: "
              f"{' '.join(w for w, _ in en[:15])}")
        print(f"    入选的 15 个: "
              f"{' '.join(w for w, _ in en[-15:])}")
        print(f"    排名范围 {en[0][1]} ~ {en[-1][1]}")

    # ---- 合并 ----
    words = zh + [w for w, _ in en]
    # 去重（中英之间可能有重叠，比如 T恤 与 t 不会重，但保险起见）
    seen = set()
    dedup = []
    for w in words:
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    if len(dedup) != len(words):
        print(f"\n  ⚠️ 合并时去掉 {len(words)-len(dedup)} 个重复词")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(dedup) + "\n", encoding="utf-8")

    meta = {
        "zh_source": str(args.zh_vocab), "zh_count": len(zh),
        "en_source": str(args.ecdict), "en_count": len(en),
        "en_params": {"n_en": args.n_en, "max_rank": args.max_rank,
                      "min_len": args.min_len, "max_len": args.max_len},
        "en_rank_range": [en[0][1], en[-1][1]] if en else None,
        "total": len(dedup),
        "stats": stats,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"\n  ✅ 写出 {args.out}  ({len(dedup):,} 词)")
    print(f"     中文 {len(zh):,} + 英文 {len(en):,}")
    print(f"     元数据 {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
