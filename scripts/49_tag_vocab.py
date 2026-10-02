# -*- coding: utf-8 -*-
"""
对混合词表做双语词性标注，并统计 feat 分布。

这是「任务 1：词表 + 词性流水线」的主脚本。

输出：
    data/mixed_tags.tsv      每行: 词 <TAB> 语言 <TAB> 标签 <TAB> feat(hex)
    data/mixed_tags.stats.json  详细统计

用法：
    python scripts/49_tag_vocab.py
    python scripts/49_tag_vocab.py --vocab data/mixed_vocab.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import bilingual_pos as bp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="混合词表双语标注")
    ap.add_argument("--vocab", type=Path, default=ROOT / "data/mixed_vocab.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "data/mixed_tags.tsv")
    args = ap.parse_args()

    if not args.vocab.exists():
        print(f"缺少 {args.vocab}")
        return 1

    words = [l.strip() for l in
             args.vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("=" * 74)
    print("混合词表双语标注")
    print("=" * 74)
    print(f"  词表 {args.vocab}  ({len(words):,} 词)")

    t0 = time.time()
    tagged = bp.tag_mixed(words)
    dt = time.time() - t0
    print(f"  标注用时 {dt:.1f}s  ({len(tagged)/max(dt,1e-6):.0f} 词/秒)")

    # ---- 写出 ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("word\tlang\ttag\tfeat\n")
        for w, t, feat in tagged:
            f.write(f"{w}\t{bp.detect_lang(w)}\t{t}\t0x{feat.value:X}\n")
    print(f"  已写出 {args.out}")

    # ---- 统计 ----
    lang_cnt = Counter()
    tag_cnt: dict[str, Counter] = defaultdict(Counter)
    slot_cnt: dict[str, Counter] = defaultdict(Counter)
    for w, t, feat in tagged:
        lang = bp.detect_lang(w)
        lang_cnt[lang] += 1
        tag_cnt[lang][t] += 1
        slot_cnt[lang][feat] += 1

    print()
    print("=" * 74)
    print("feat 槽位分布")
    print("=" * 74)
    print(f"  {'槽位':<6}{'说明':<34}{'中文':>8}{'英文':>8}{'合计':>8}")
    print("  " + "-" * 66)
    for slot in range(16):
        f = bp.Feat(slot)
        cz = slot_cnt["zh"].get(f, 0)
        ce = slot_cnt["en"].get(f, 0)
        if cz == 0 and ce == 0:
            continue
        print(f"  0x{slot:X}   {bp.feat_name(f):<32}{cz:>8,}{ce:>8,}{cz+ce:>8,}")

    total = len(tagged)
    empty_slots = [f"0x{s:X}" for s in range(16)
                   if slot_cnt["zh"].get(bp.Feat(s), 0) == 0
                   and slot_cnt["en"].get(bp.Feat(s), 0) == 0]
    print()
    if empty_slots:
        print(f"  ⚠️ 未被使用的槽位: {', '.join(empty_slots)}")
    else:
        print("  ✅ 16 个槽位全部被使用")

    # ---- 各语言的标签详情 ----
    for lang in ("zh", "en"):
        print()
        print("=" * 74)
        print(f"{'中文' if lang == 'zh' else '英文'}标签分布"
              f"（{lang_cnt[lang]:,} 词，{len(tag_cnt[lang])} 种标签）")
        print("=" * 74)
        print(f"  {'标签':<10}{'数量':>7}{'占比':>8}{'feat':<7}示例")
        print("  " + "-" * 68)
        for tag, n in tag_cnt[lang].most_common(20):
            feat = (bp.pos_to_feat_zh(tag) if lang == "zh"
                    else bp.pos_to_feat_en(tag))
            ex = [w for w, t, _ in tagged
                  if t == tag and bp.detect_lang(w) == lang][:5]
            print(f"  {tag:<10}{n:>7,}{n/lang_cnt[lang]*100:>7.1f}%  "
                  f"0x{feat.value:X}    {'、'.join(ex)}")

        # 未映射到已知槽位的标签（FALLBACK）
        fb = [(t, n) for t, n in tag_cnt[lang].items()
              if (bp.pos_to_feat_zh(t) if lang == "zh"
                  else bp.pos_to_feat_en(t)) == bp.Feat.FALLBACK]
        if fb:
            print(f"\n  ⚠️ 落入 FALLBACK 的标签（{len(fb)} 种，"
                  f"{sum(n for _, n in fb):,} 词）:")
            for t, n in sorted(fb, key=lambda x: -x[1])[:12]:
                ex = [w for w, tt, _ in tagged if tt == t][:4]
                print(f"      {t!r:<14} {n:>5}   {'、'.join(ex)}")

    stats = {
        "vocab": str(args.vocab), "total": total, "seconds": dt,
        "lang_counts": dict(lang_cnt),
        "tag_counts": {k: dict(v) for k, v in tag_cnt.items()},
        "slot_counts": {k: {f"0x{s:X}": n for s, n in v.items()}
                        for k, v in slot_cnt.items()},
        "empty_slots": empty_slots,
    }
    sp = args.out.with_suffix(".stats.json")
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"\n  统计 -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
