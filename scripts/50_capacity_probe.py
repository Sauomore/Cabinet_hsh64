# -*- coding: utf-8 -*-
"""
算清楚 HSH-64 的容量上限，以及当前 6,109 词的实际分桶情况。

为什么必须先算：
    `abs` 只有 8 bit，每个 (feat, sim_low8) 桶最多容纳 256 个词。
    桶 = feat(16) × sim_low8(256) = 4,096 个。
    理论总容量 4,096 × 256 = 1,048,576，但实际取决于两件事：
      1. sim_low8 是否均匀（DeepHash 输出的低 8 位）
      2. feat 是否均匀（词性分布，实测很偏 —— 名词+形容词占 86%）
    分布越偏，可用规模越小。盲目扩到 5 万词可能撞上 256 上限。

输出：
    results/capacity_probe.json

用法：
    python scripts/50_capacity_probe.py
    python scripts/50_capacity_probe.py --model data/mixed_deep_hash_52.bin \
        --embedding data/mixed_embedding.cache --tags data/mixed_tags.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from train_deep_hash_v3_64 import (  # noqa: E402
    load_deep_hash_model,
    parse_embedding_cache,
)


def load_tags(path: Path) -> dict[str, int]:
    """读 mixed_tags.tsv: word<TAB>lang<TAB>tag<TAB>0xFEAT"""
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 4:
            try:
                out[p[0]] = int(p[3], 16)
            except ValueError:
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="HSH-64 容量探针")
    ap.add_argument("--embedding", type=Path,
                    default=ROOT / "data/mixed_embedding.cache")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "data/mixed_deep_hash_52.bin")
    ap.add_argument("--tags", type=Path, default=ROOT / "data/mixed_tags.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "results/capacity_probe.json")
    args = ap.parse_args()

    print("=" * 74)
    print("HSH-64 容量探针")
    print("=" * 74)

    # ---- 1. 读 embedding ----
    dim, words, vecs = parse_embedding_cache(args.embedding)
    print(f"\n  embedding  {len(words):,} 词 × {dim} 维")

    # ---- 2. 读 DeepHash ----
    mean, net = load_deep_hash_model(args.model)
    n_bits = net.fc2.out_features
    hidden = net.fc1.out_features
    print(f"  deep hash  dim={dim} hidden={hidden} n_bits={n_bits}")

    # ---- 3. 编码（x - mean -> fc2(relu(bn1(fc1(x))))) ----
    with torch.no_grad():
        x = torch.from_numpy(vecs.astype(np.float32)) - torch.from_numpy(mean)
        logits = net(x)
        bits = (logits > 0).to(torch.int64)
        sims = torch.zeros(len(words), dtype=torch.int64)
        for b in range(n_bits):
            sims |= bits[:, b] << (n_bits - 1 - b)
        sims = sims.numpy()

    # ---- 4. feat ----
    tag_map = load_tags(args.tags)
    feats = np.array([tag_map.get(w, 0xF) for w in words], dtype=np.int64)
    sim_low = (sims & 0xFF).astype(np.int64)

    print(f"\n  编码完成 {len(words):,} 词 × {n_bits} bit")
    print(f"  sim 唯一码 {len(np.unique(sims)):,} / {len(words):,} "
          f"= {len(np.unique(sims))/len(words):.2%}")
    print(f"  feat 覆盖  {len(np.unique(feats))} / 16 槽")

    # ---- 5. feat 分布 ----
    fc = Counter(feats.tolist())
    print("\n  feat 分布（偏斜会压缩有效容量）:")
    for f, c in fc.most_common(10):
        print(f"    0x{f:X}  {c:>6,}  ({c/len(words)*100:>5.1f}%)")

    # ---- 6. 桶大小 ----
    buckets = np.stack([feats, sim_low], axis=1)
    bc = Counter(map(tuple, buckets))
    sizes = np.array(list(bc.values()))
    cap_total = 16 * 256

    print("\n  (feat, sim_low8) 分桶:")
    print(f"    总桶数        {cap_total:,}")
    print(f"    非空桶        {len(bc):,}  ({len(bc)/cap_total:.1%})")
    print(f"    平均桶大小    {sizes.mean():.2f}")
    print(f"    中位数        {np.median(sizes):.0f}")
    print(f"    P99           {np.percentile(sizes, 99):.0f}")
    print(f"    最大桶        {sizes.max()}")
    print(f"    超 256 的桶   {(sizes > 256).sum()}  ← 完美哈希退化为最小碰撞")

    # 只看最大的 feat（容量瓶颈在这里）
    worst_feat = fc.most_common(1)[0][0]
    mask = feats == worst_feat
    sub = Counter(sim_low[mask].tolist())
    sub_sizes = np.array(list(sub.values()))
    print(f"\n  最大 feat 槽 0x{worst_feat:X} 的内部情况（{mask.sum():,} 词）:")
    print(f"    占用 sim_low 桶  {len(sub)} / 256")
    print(f"    平均桶           {sub_sizes.mean():.2f}")
    print(f"    最大桶           {sub_sizes.max()}")

    # ---- 7. 外推 ----
    slope = sizes.max() / len(words)
    print("\n" + "=" * 74)
    print("容量外推（线性假设：分桶形状不变，按词数等比放大）")
    print("=" * 74)
    print(f"  {'词表规模':>10}{'预期最大桶':>13}{'余量':>10}{'判定':>14}")
    print("  " + "-" * 49)
    for n in (3_109, 6_109, 20_000, 50_000, 100_000, 200_000):
        pred = slope * n
        margin = 256 - pred
        verdict = "✅ 安全" if pred < 200 else ("⚠️ 接近上限" if pred < 256 else "❌ 撞上限")
        print(f"  {n:>10,}{pred:>13.0f}{margin:>10.0f}{verdict:>14}")

    print("\n  注意：这是线性外推，实际扩张时 sim_low8 会重新分布。")
    print("        但 feat 的偏斜（最大槽占 "
          f"{fc.most_common(1)[0][1]/len(words):.0%}）会持续存在。")
    print("        若外推显示会撞上限，办法依次是：")
    print("          1. 给 feat 重新分槽（把名词拆细 → 16 槽用满）")
    print("          2. 把 abs 从 8 bit 扩到更多位（改位布局，破坏兼容）")
    print("          3. 接受 abs 碰撞 —— 它不参与 MIH 检索，只影响精确码相等判定")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_words": len(words), "n_bits": int(n_bits), "dim": int(dim),
        "hidden_dim": int(hidden),
        "sim_unique": int(len(np.unique(sims))),
        "feat_counts": {f"0x{f:X}": int(c) for f, c in fc.items()},
        "buckets_total": cap_total,
        "buckets_nonempty": int(len(bc)),
        "bucket_mean": float(sizes.mean()),
        "bucket_p99": float(np.percentile(sizes, 99)),
        "bucket_max": int(sizes.max()),
        "bucket_over_256": int((sizes > 256).sum()),
        "largest_feat_slot": f"0x{worst_feat:X}",
        "largest_feat_words": int(mask.sum()),
        "largest_feat_max_bucket": int(sub_sizes.max()),
        "slope_max_bucket_per_word": float(slope),
        "extrapolation": {str(n): float(slope * n)
                          for n in (3109, 6109, 20000, 50000, 100000, 200000)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
