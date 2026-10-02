# -*- coding: utf-8 -*-
"""
快速验证：跨语言语义对齐是否成立？

为什么要先做这个：
    「混合索引」（中英词混在一个码本里）成立的前提是——
    embedding 必须把「苹果」和「apple」放在相近的位置。
    这件事不能假设。

    而且 bge-m3 有 2.27 GB，本机到 huggingface.co 只有 ~270 KB/s，
    要下 2.4 小时。所以在下载前，先用【已有】的模型测一个初步信号。

本脚本做三件事：
    1. 用已有的 bge-small-zh 测中文词对的相似度结构（基线）
    2. 用同一个中文模型测英文词对——看它是否仍保留语义结构
    3. 测中英对照词对——看跨语言信号有多强

⚠️ 重要限制：
    bge-small-zh 是【中文专用】模型，它给英文输出的向量不可靠。
    这个测试只能说明「中文模型对英文输入的部分行为」，
    不能替代 bge-m3 的真实跨语言能力测试。
    它的作用是：如果连基本结构都没有，说明更大问题；
    如果有一些结构，说明值得投入下载 bge-m3。

用法：
    python scripts/43_probe_crosslingual.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

#: 中英对照词对（人工挑选，覆盖不同词类）
PAIRS = [
    ("苹果", "apple"), ("香蕉", "banana"), ("猫", "cat"), ("狗", "dog"),
    ("水", "water"), ("火", "fire"), ("太阳", "sun"), ("月亮", "moon"),
    ("山", "mountain"), ("河", "river"), ("树", "tree"), ("花", "flower"),
    ("学校", "school"), ("老师", "teacher"), ("学生", "student"),
    ("医生", "doctor"), ("医生", "physician"), ("书", "book"),
    ("计算机", "computer"), ("手机", "phone"), ("汽车", "car"),
    ("飞机", "airplane"), ("火车", "train"), ("北京", "Beijing"),
    ("上海", "Shanghai"), ("中国", "China"), ("美国", "America"),
    ("爱", "love"), ("恨", "hate"), ("快乐", "happy"), ("悲伤", "sad"),
    ("大", "big"), ("小", "small"), ("快", "fast"), ("慢", "slow"),
    ("热", "hot"), ("冷", "cold"), ("新", "new"), ("旧", "old"),
]

#: 负对照：明确不相关的词对
NEG_PAIRS = [
    ("苹果", "mountain"), ("猫", "计算机"), ("水", "老师"),
    ("太阳", "悲伤"), ("书", "飞机"), ("爱", "石头"),
]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    import argparse
    import torch
    from sentence_transformers import SentenceTransformer

    ap = argparse.ArgumentParser(description="跨语言语义对齐探针")
    ap.add_argument("--model", default=r"F:\bge-m3",
                    help="模型路径或名称（默认本地 bge-m3）")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/crosslingual_probe.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    print("跨语言语义对齐探针")
    print("=" * 76)

    mpath = Path(args.model)
    model = SentenceTransformer(str(mpath) if mpath.exists() else args.model,
                               device=device)
    is_m3 = "m3" in str(args.model).lower()
    print(f"  模型 {args.model}")
    print(f"  维度 {model.get_sentence_embedding_dimension()}   设备 {device}")
    if not is_m3:
        print("  ⚠️ 这是中文专用模型，英文结果仅供初步参考")
    else:
        print("  ✅ 多语言模型，跨语言结果是正式的")

    zh = [p[0] for p in PAIRS]
    en = [p[1] for p in PAIRS]

    vz = model.encode(zh, normalize_embeddings=True, show_progress_bar=False)
    ve = model.encode(en, normalize_embeddings=True, show_progress_bar=False)

    # ---- 1. 跨语言对照 ----
    print("\n" + "=" * 76)
    print("1. 中英对照词对的相似度（正样本）")
    print("=" * 76)
    pos = [cos(vz[i], ve[i]) for i in range(len(PAIRS))]

    # ---- 2. 负对照 ----
    print("\n" + "=" * 76)
    print("2. 不相关词对（负样本）")
    print("=" * 76)
    neg = []
    for a, b in NEG_PAIRS:
        va = model.encode([a], normalize_embeddings=True)[0]
        vb = model.encode([b], normalize_embeddings=True)[0]
        neg.append(cos(va, vb))

    # ---- 3. 中文内部相似度（基线） ----
    print("\n" + "=" * 76)
    print("3. 中文之间的相似度（基线，看模型本身是否正常）")
    print("=" * 76)
    zh_pairs = [("苹果", "香蕉"), ("苹果", "猫"), ("学校", "老师"),
                ("太阳", "月亮"), ("大", "小")]
    zh_sim = []
    for a, b in zh_pairs:
        va = model.encode([a], normalize_embeddings=True)[0]
        vb = model.encode([b], normalize_embeddings=True)[0]
        s = cos(va, vb)
        zh_sim.append(s)
        print(f"    {a} vs {b:<8}  {s:+.4f}")

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    print("判别")
    print("=" * 76)
    mp, mn = np.mean(pos), np.mean(neg)
    print(f"  正样本（中英对照）  均值 {mp:+.4f}   "
          f"范围 [{min(pos):+.4f}, {max(pos):+.4f}]")
    print(f"  负样本（不相关）    均值 {mn:+.4f}   "
          f"范围 [{min(neg):+.4f}, {max(neg):+.4f}]")
    print(f"  分离度 (正 - 负)    {mp - mn:+.4f}")
    print()

    # 逐对列出，便于人工检查
    print("  逐对明细（正样本）:")
    order = np.argsort(pos)[::-1]
    for i in order:
        marker = "  " if pos[i] > mp else "↓ "
        print(f"    {marker}{PAIRS[i][0]:<10} {PAIRS[i][1]:<12} {pos[i]:+.4f}")

    print("\n  逐对明细（负样本）:")
    for (a, b), s in zip(NEG_PAIRS, neg):
        print(f"    {a:<10} {b:<12} {s:+.4f}")

    print("\n" + "=" * 76)
    if mp - mn > 0.15:
        verdict = ("有跨语言信号 —— 值得投入下载 bge-m3 做正式测试")
    elif mp - mn > 0.05:
        verdict = ("信号很弱 —— bge-m3 可能好很多（它是多语言的），但需要正式测")
    else:
        verdict = ("几乎没有跨语言信号 —— 需要重新考虑方案")
    print(f"  初步结论: {verdict}")
    print("=" * 76)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": "BAAI/bge-small-zh-v1.5",
        "note": "中文专用模型，英文仅供初步参考",
        "pos_mean": mp, "neg_mean": mn, "separation": mp - mn,
        "pos": [{"zh": PAIRS[i][0], "en": PAIRS[i][1], "cos": pos[i]}
                for i in range(len(PAIRS))],
        "neg": [{"zh": a, "en": b, "cos": s}
                for (a, b), s in zip(NEG_PAIRS, neg)],
        "zh_baseline": [{"a": a, "b": b, "cos": s}
                        for (a, b), s in zip(zh_pairs, zh_sim)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
