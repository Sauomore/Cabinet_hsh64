# -*- coding: utf-8 -*-
"""
导出 Python 侧的 (标签 → feat) 映射表，供 Rust 对拍。

为什么需要：
    中英双语的 feat 映射表现在存在于两个地方：
      · Python: scripts/bilingual_pos.py  (建码本时用)
      · Rust:   src/pos_map.rs            (查询码本时用)
    两边必须完全一致。不一致的后果是「Python 建的码本，Rust 查出来不对」——
    这类 bug 很难发现，因为单看任何一边都正常。

    所以导出成 JSON，让 Rust 测试逐项断言。

输出格式：
    {
      "zh": {"n": 0, "nr": 0, ...},        # jieba 标签 -> feat 值
      "en": {"NN": 0, "VB": 1, ...},       # PTB 标签 -> feat 值
      "feat_names": {"0": "名词/noun", ...}
    }

用法：
    python scripts/46_export_feat_table.py
    python scripts/46_export_feat_table.py --out tests/data/feat_table.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import bilingual_pos as bp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 feat 映射表供 Rust 对拍")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests/data/feat_table.json")
    args = ap.parse_args()

    payload = {
        "_comment": "由 scripts/46_export_feat_table.py 生成，"
                    "Rust 侧由 tests/feat_parity.rs 校验。改动任一侧都要重新导出。",
        "zh": {t: int(f) for t, f in sorted(bp._ZH.items())},
        "en": {t: int(f) for t, f in sorted(bp._EN.items())},
        "feat_names": {str(int(f)): bp.feat_name(f) for f in bp.Feat},
        "slots": {f.name: int(f) for f in bp.Feat},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("=" * 66)
    print("导出 feat 映射表（Python -> JSON）")
    print("=" * 66)
    print(f"  输出 {args.out}")
    print(f"  中文标签 {len(payload['zh'])} 个")
    print(f"  英文标签 {len(payload['en'])} 个")
    print(f"  槽位名称 {len(payload['feat_names'])} 个")
    print()
    print("  中文标签分布:")
    from collections import Counter
    cz = Counter(payload["zh"].values())
    for slot in sorted(cz):
        names = [t for t, v in payload["zh"].items() if v == slot]
        print(f"    0x{slot:X}  {bp.feat_name(bp.Feat(slot)):<34} "
              f"{len(names):>3} 个: {' '.join(names[:8])}"
              + (" …" if len(names) > 8 else ""))
    print()
    print("  英文标签分布:")
    ce = Counter(payload["en"].values())
    for slot in sorted(ce):
        names = [t for t, v in payload["en"].items() if v == slot]
        print(f"    0x{slot:X}  {bp.feat_name(bp.Feat(slot)):<34} "
              f"{len(names):>3} 个: {' '.join(names[:8])}"
              + (" …" if len(names) > 8 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
