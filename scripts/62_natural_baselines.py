# -*- coding: utf-8 -*-
"""
在自然词表上补基线：PCA-52 与随机码，用来判断 HSH-64 的 0.608 到底算不算好。

为什么必须补：
    之前的规模评估只比较了「HSH-64 在不同规模/词表上的自己」，
    没有比较「HSH-64 与最简基线」。
    0.608 这个数字单独看没有意义 —— 如果 PCA-52 也能拿到 0.55，
    那 HSH-64 的增益就很有限；如果 PCA-52 只有 0.2，增益就是实质性的。

    论文在合成词表上报告 PCA-52 = 0.5318，本脚本先在合成词表上复现它，
    确认实现正确，再套到自然词表上。

三种码：
    随机 52 bit        —— 地板
    PCA-52 sign 量化   —— 论文的纯 HSH 基线
    HSH-64（已训练）    —— 读 data/eval 的产物

用法：
    python scripts/62_natural_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from eval_helpers import load_exported, load_pca_student  # noqa: E402
from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache  # noqa: E402
from cabinet_bnn.hsh.deep_hash import project, quantize_bits  # noqa: E402
from cabinet_bnn.hsh.post_optimize import (  # noqa: E402
    hamming_matrix,
    load_override_cache,
    recall_at_k_from_ranking,
)

EVAL = ROOT / "data/eval"


def recall_overlap(bits: np.ndarray, teacher_sim: np.ndarray,
                   k: int = 10) -> float:
    """overlap@k：真值 = teacher top-k，检索 = Hamming top-k（论文口径）。"""
    n = bits.shape[0]
    kk = min(k, n - 1)
    pos = np.argsort(-teacher_sim, axis=1)[:, 1:kk + 1]
    ham = hamming_matrix(bits)
    return float(recall_at_k_from_ranking(pos, ham, kk))


def pca52_codes(teacher: np.ndarray, n_bits: int = 52) -> np.ndarray:
    """PCA 降到 n_bits 维，再按符号量化成位。

    这就是论文表格里的「PCA-52（纯 HSH）」基线：
    不做任何学习，只用 PCA 主成分的符号作为 52 位码。
    """
    mean = teacher.mean(axis=0, keepdims=True)
    xc = teacher - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[:n_bits].T                 # (N, 52) 连续值
    # 逐维减去中位数再取符号，避免全落在同一侧
    proj = proj - np.median(proj, axis=0, keepdims=True)
    return (proj >= 0).astype(np.uint8)


def load_teacher(vocab: Path, cache: Path):
    words = [l.strip() for l in vocab.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    dim, tw, teacher = read_cache(cache)
    if tw != words:
        idx = {w: i for i, w in enumerate(tw)}
        teacher = teacher[[idx[w] for w in words]]
    return words, teacher


def hsh_codes(words: list[str], teacher: np.ndarray, tag: str,
              use_override: bool) -> np.ndarray | None:
    """载入已训练的 HSH-64 产物，返回 52 位码。"""
    mp = EVAL / f"deep_hash_{tag}.bin"
    if not mp.exists():
        return None
    student = load_pca_student(teacher, 512, EVAL / f"pca512_{tag}.npz")
    mean, model = load_exported(mp)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    bits = quantize_bits(project(model, mean, student, device=dev))
    if use_override:
        ov = EVAL / f"sim_override_{tag}.bin"
        if ov.exists():
            ow, obits = load_override_cache(ov)
            idx = {w: i for i, w in enumerate(ow)}
            out = np.zeros_like(bits)
            for i, w in enumerate(words):
                if w in idx:
                    out[i] = obits[idx[w]]
            return out
    return bits


def main() -> int:
    ap = argparse.ArgumentParser(description="自然词表基线对比")
    ap.add_argument("--sizes", type=int, nargs="+", default=[3000, 16000])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/natural_baselines.json")
    args = ap.parse_args()

    print("=" * 84)
    print("自然词表上的基线对比（回答：HSH-64 的 0.6 算不算好）")
    print("=" * 84)

    results = []

    # ---------- 先验证 PCA-52 实现在合成词表上能复现 0.5318 ----------
    syn_vocab = ROOT / "tests/data/vocab_3109.txt"
    syn_cache = Path(r"F:\Cabinet-BNN\results\bge_large_3109.cache")
    if syn_vocab.exists() and syn_cache.exists():
        print("\n[校验] 合成 3,109 词表 —— 论文 PCA-52 = 0.5318")
        words, teacher = load_teacher(syn_vocab, syn_cache)
        tn = normalize_rows(teacher)
        ts = tn @ tn.T
        rng = np.random.default_rng(args.seed)
        rand_bits = rng.integers(0, 2, size=(len(words), 52), dtype=np.uint8)
        pca_bits = pca52_codes(teacher, 52)
        r_rand = recall_overlap(rand_bits, ts, args.k)
        r_pca = recall_overlap(pca_bits, ts, args.k)
        hsh_raw = hsh_codes(words, teacher, "repro3109", False)
        hsh_opt = hsh_codes(words, teacher, "repro3109", True)
        r_hsh_raw = recall_overlap(hsh_raw, ts, args.k) if hsh_raw is not None else None
        r_hsh_opt = recall_overlap(hsh_opt, ts, args.k) if hsh_opt is not None else None

        print(f"    随机 52 bit            {r_rand:.4f}   （理论地板 10/3109 = 0.0032）")
        print(f"    PCA-52                 {r_pca:.4f}   论文 0.5318 "
              f"→ 差 {r_pca - 0.5318:+.4f}")
        if r_hsh_raw is not None:
            print(f"    HSH-64 裸码            {r_hsh_raw:.4f}   论文 0.4265")
        if r_hsh_opt is not None:
            print(f"    HSH-64 + 后处理        {r_hsh_opt:.4f}   论文 0.7404")
        results.append({
            "name": "synthetic_3109", "n": len(words),
            "random": r_rand, "pca52": r_pca,
            "hsh_raw": r_hsh_raw, "hsh_post": r_hsh_opt,
            "paper_pca52": 0.5318, "paper_hsh_raw": 0.4265,
            "paper_hsh_post": 0.7404,
        })

    # ---------- 自然词表 ----------
    print()
    print(f"  {'词表':<14}{'N':>7}{'随机':>9}{'PCA-52':>9}"
          f"{'HSH裸码':>10}{'HSH+后处理':>12}{'vs PCA':>10}")
    print("  " + "-" * 72)

    for size in args.sizes:
        vocab = ROOT / f"data/natural_vocab_{size}.txt"
        cache = EVAL / f"teacher_nat{size}.cache"
        if not vocab.exists() or not cache.exists():
            print(f"  nat{size:<11}  缺文件，跳过")
            continue
        words, teacher = load_teacher(vocab, cache)
        tn = normalize_rows(teacher)
        ts = tn @ tn.T
        n = len(words)

        rng = np.random.default_rng(args.seed)
        rand_bits = rng.integers(0, 2, size=(n, 52), dtype=np.uint8)
        pca_bits = pca52_codes(teacher, 52)

        r_rand = recall_overlap(rand_bits, ts, args.k)
        r_pca = recall_overlap(pca_bits, ts, args.k)
        hsh_raw = hsh_codes(words, teacher, f"nat{size}", False)
        hsh_opt = hsh_codes(words, teacher, f"nat{size}", True)
        r_raw = recall_overlap(hsh_raw, ts, args.k) if hsh_raw is not None else float("nan")
        r_opt = recall_overlap(hsh_opt, ts, args.k) if hsh_opt is not None else float("nan")

        gain = r_opt - r_pca
        print(f"  nat{size:<11}{n:>7,}{r_rand:>9.4f}{r_pca:>9.4f}"
              f"{r_raw:>10.4f}{r_opt:>12.4f}{gain:>+10.4f}")
        results.append({
            "name": f"natural_{size}", "n": n,
            "random": r_rand, "pca52": r_pca,
            "hsh_raw": None if np.isnan(r_raw) else r_raw,
            "hsh_post": None if np.isnan(r_opt) else r_opt,
            "hsh_minus_pca": None if np.isnan(gain) else gain,
        })

    # ---------- 结论 ----------
    print()
    print("=" * 84)
    print("解读")
    print("=" * 84)
    nat = [r for r in results if r["name"].startswith("natural_")]
    for r in nat:
        if r["hsh_post"] is None:
            continue
        ratio = r["hsh_post"] / max(r["pca52"], 1e-9)
        print(f"  {r['name']}: HSH-64 {r['hsh_post']:.4f} / PCA-52 {r['pca52']:.4f}"
              f" = {ratio:.2f}x   （绝对增益 {r['hsh_minus_pca']:+.4f}）")
    print()
    if nat:
        gains = [r["hsh_minus_pca"] for r in nat if r["hsh_minus_pca"] is not None]
        if gains and min(gains) > 0.15:
            print("  >> HSH-64 相对最简基线的增益是实质性的（> 0.15）。")
            print("     换句话说：论文词表「偏简单」这个问题削弱的是绝对数字的可比性，")
            print("     不是方法相对基线的优势。")
        elif gains and min(gains) > 0.05:
            print("  >> 增益存在但不算大（0.05 ~ 0.15）。")
            print("     结论要更保守：HSH-64 的优势部分是「训练 + 后处理」带来的，")
            print("     不完全是编码方案本身。")
        elif gains:
            print("  >> 增益有限（< 0.05），需要重新审视方法的必要性。")
    print()
    print("  注意：PCA-52 不做任何学习，也不用后处理；")
    print("       HSH-64 用了 500 epochs 训练 + 10 轮贪心翻转。")
    print("       所以这个对比是「整套流程 vs 零成本基线」，不是「编码方案 vs 编码方案」。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
