# -*- coding: utf-8 -*-
"""汇总 PQ 基线：按【总内存】而非「每项字节数」排，并画帕累托图。

为什么按总内存：PQ 有固定码本开销（512 KB 或 1024 KB），在 3k~16k 词表的规模下
码本远大于码本身。嵌入式场景关心的是总占用，不是每项字节数。
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

hsh = {  # 记录值（本轮已用覆盖表逐位复现，差 0.0000）
    "nat3000": 0.6082, "nat6000": 0.5990, "nat10000": 0.5936,
    "nat16000": 0.5965, "mixed6109": 0.6086,
}
hsh_n = {"nat3000": 3000, "nat6000": 6000, "nat10000": 10000,
         "nat16000": 16000, "mixed6109": 6109}

d8 = json.load(io.open(R / "pq_baseline.json", encoding="utf-8"))
dc = json.load(io.open(R / "pq_budget_curve.json", encoding="utf-8"))

datasets = {}
for src in (d8, dc):
    for tag, rec in src["datasets"].items():
        datasets.setdefault(tag, {"n": rec["n"], "pq": {}, "pca512_float": rec.get("pca512_float")})
        datasets[tag]["pq"].update(rec["pq"])

print("=" * 92)
print("  PQ 基线汇总（overlap@10，口径已用覆盖表逐位复现）")
print("=" * 92)

summary = {"hsh_record": hsh, "datasets": {}}

for tag in ("nat3000", "nat6000", "nat10000", "nat16000", "mixed6109"):
    if tag not in datasets:
        continue
    d = datasets[tag]
    n = d["n"]
    print(f"\n[{tag}]  n={n}    HSH-64: {hsh[tag]:.4f} ({n*8/1024:.0f} KB, 零码本)")
    rows = [("HSH-64", 8, 0, hsh[tag], "Hamming+后处理")]
    for key, v in sorted(d["pq"].items()):
        iname, b = key.split("_pq")
        nb = int(b.rstrip("B"))
        rows.append((f"PQ-{nb}B@{iname}", nb, v["codebook_bytes"], v["recall"], ""))
    if d["pca512_float"]:
        rows.append(("PCA-512 浮点", 2048, 0, d["pca512_float"]["recall"], "不量化"))
    rows.sort(key=lambda r: n * r[1] + r[2])
    print(f"    {'方法':<18}{'码B/项':>8}{'码本KB':>9}{'总内存KB':>10}{'recall@10':>11}  备注")
    print("    " + "-" * 84)
    for name, nb, cb, rec, note in rows:
        tot = (n * nb + cb) / 1024
        mark = "  ←" if name == "HSH-64" else ""
        print(f"    {name:<18}{nb:>8}{cb/1024:>9.0f}{tot:>10.0f}{rec:>11.4f}  {note}{mark}")
    summary["datasets"][tag] = {
        "n": n, "hsh": hsh[tag], "hsh_total_kb": n * 8 / 1024,
        "rows": [{"method": r[0], "bytes_per_item": r[1], "codebook_bytes": r[2],
                  "total_kb": (n * r[1] + r[2]) / 1024, "recall": r[3]} for r in rows],
    }

R.joinpath("pq_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
print(f"\n  已写入 {R / 'pq_summary.json'}")

# ---------------------------------------------------------------- 帕累托图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, tag in zip(axes, ("nat3000", "nat16000")):
        d = summary["datasets"][tag]
        for iname, color in (("pca512", "#58a6ff"), ("teacher1024", "#bc8cff")):
            pts = [(r["total_kb"], r["recall"]) for r in d["rows"]
                   if r["method"].startswith("PQ") and iname in r["method"]]
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=color, label=f"PQ on {iname}")
        ax.scatter([d["hsh_total_kb"]], [d["hsh"]], s=200, marker="*",
                   color="#3fb950", zorder=5, label="HSH-64 (8 B, no codebook)")
        pf = [r for r in d["rows"] if r["method"].startswith("PCA")]
        if pf:
            ax.scatter([pf[0]["total_kb"]], [pf[0]["recall"]], s=90, marker="s",
                       color="#d29922", zorder=5, label="PCA-512 float (2 KB/item)")
        ax.set_xscale("log")
        ax.set_xlabel("Total memory (KB, log scale)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"{tag}  (n={d['n']})")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.34, 0.90)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = R / "pq_pareto.png"
    fig.savefig(out, dpi=150)
    print(f"  已写入 {out}")
except Exception as e:
    print(f"  绘图跳过：{type(e).__name__}: {e}")
