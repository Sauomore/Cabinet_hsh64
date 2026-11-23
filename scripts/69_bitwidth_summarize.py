# -*- coding: utf-8 -*-
"""合并两次码宽扫描，画曲线并计算后处理增益。"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path(__file__).resolve().parent.parent / "results"

low = json.load(io.open(R / "bitwidth_sweep_low.json", encoding="utf-8"))
high = json.load(io.open(R / "bitwidth_sweep_high.json", encoding="utf-8"))
res = {**low["results"], **high["results"]}
widths = sorted(int(w) for w in res)
seeds = low["seeds"]

merged = {"tag": low["tag"], "n": low["n"], "widths": widths, "seeds": seeds,
          "results": {str(w): res[str(w)] for w in widths}}
(R / "bitwidth_sweep.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

print("=" * 96)
print(f"  码宽扫描汇总 —— {low['tag']}  n={low['n']}  DH 为 {len(seeds)} 种子均值")
print("=" * 96)

hdr = "".join(f"{w:>9}" for w in widths)
print(f"    {'方法':<18}{hdr}")
rows = [
    ("PCA 裸码", "pca_bare"),
    ("PCA + 后处理", "pca_post"),
    ("DH 裸码", "deephash_bare"),
    ("DH + 后处理", "deephash_post"),
]
for name, key in rows:
    cells = []
    for w in widths:
        v = res[str(w)][key]
        cells.append(f"{v:.4f}" if isinstance(v, float) else f"{v['mean']:.4f}")
    print(f"    {name:<18}" + "".join(f"{c:>9}" for c in cells))

print()
print("=" * 96)
print("  后处理增益（后处理 − 裸码）")
print("=" * 96)
print(f"    {'':<18}{hdr}")
gp = [res[str(w)]["pca_post"] - res[str(w)]["pca_bare"] for w in widths]
gd = [res[str(w)]["deephash_post"]["mean"] - res[str(w)]["deephash_bare"]["mean"] for w in widths]
print(f"    {'PCA':<18}" + "".join(f"{v:>+9.4f}" for v in gp))
print(f"    {'DeepHash':<18}" + "".join(f"{v:>+9.4f}" for v in gd))

print()
print("=" * 96)
print("  逐位宽胜者（后处理后）")
print("=" * 96)
for w in widths:
    r = res[str(w)]
    p, d = r["pca_post"], r["deephash_post"]["mean"]
    win = "PCA" if p > d else "DH"
    print(f"    {w:>3} 位   PCA {p:.4f}   DH {d:.4f}   → {win} 领先 {abs(p - d):.4f}"
          f"   唯一码率 DH {r['deephash_unique_rate']['mean']:.2%}")

# ---------------------------------------------------------------- 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.2, 5.0))
xs = widths
pca_bare = [res[str(w)]["pca_bare"] for w in xs]
pca_post = [res[str(w)]["pca_post"] for w in xs]
dh_bare = np.array([res[str(w)]["deephash_bare"]["mean"] for w in xs])
dh_bare_sd = np.array([res[str(w)]["deephash_bare"]["std"] for w in xs])
dh_post = np.array([res[str(w)]["deephash_post"]["mean"] for w in xs])
dh_post_sd = np.array([res[str(w)]["deephash_post"]["std"] for w in xs])

ax.plot(xs, pca_bare, "o--", color="#d29922", label="PCA+sign, bare")
ax.plot(xs, pca_post, "o-", color="#d29922", label="PCA+sign, + post-processing")
ax.errorbar(xs, dh_bare, yerr=dh_bare_sd, fmt="s--", color="#58a6ff",
            capsize=3, label="DeepHash, bare")
ax.errorbar(xs, dh_post, yerr=dh_post_sd, fmt="s-", color="#58a6ff",
            capsize=3, label="DeepHash, + post-processing")

ax.axvline(16, color="#f85149", ls=":", lw=1.2)
ax.axvline(40, color="#f85149", ls=":", lw=1.2)
ax.text(16, 0.03, " crossover\n after post", fontsize=7, color="#f85149", ha="center")
ax.text(40, 0.03, " crossover\n bare", fontsize=7, color="#f85149", ha="center")

ax.set_xlabel("sim bit width")
ax.set_ylabel("Recall@10 (overlap)")
ax.set_title(f"Bit-width sweep on synthetic 3,109 (bge-small -> bge-large)")
ax.set_xticks(xs)
ax.set_ylim(0, 0.85)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
out = R / "bitwidth_sweep.png"
fig.savefig(out, dpi=150)
print(f"\n  已写入 {out} 与 {R / 'bitwidth_sweep.json'}")
