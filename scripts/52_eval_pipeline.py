# -*- coding: utf-8 -*-
"""
HSH-64 统一评估流水线：词表 → 教师嵌入 → PCA 学生 → DeepHash → 后处理 → Recall。

设计目标：
    1. 一个脚本跑完论文的完整流程，用于复现与规模化实验
    2. 每一步的产物都落盘，并且【Rust 侧可以直接消费】
       —— 学生输入存成 512 维 cache，Rust 的 Encoder 能直接读
    3. 每次运行把结果追加到 results/eval_runs.jsonl，便于画规模曲线

为什么学生输入要存盘：
    论文的 h512 模型要求 512 维输入，但教师是 bge-large 的 1024 维。
    当初 02_mvp_hsh64.py 是【内存里】做 PCA-512 截断，所以 Rust 没法复现。
    这里把 PCA 基和学生缓存都存下来，Rust 就能用同一套输入。

两种模式：
    --model/--override 给定  → 只评估已有产物（复现论文基线）
    未给定                   → 训练 + 后处理 + 评估（规模化实验）

用法：
    # 复现基线（用已有的 3,109 模型）
    python scripts/52_eval_pipeline.py --vocab tests/data/vocab_3109.txt \
        --teacher-cache F:/Cabinet-BNN/results/bge_large_3109.cache \
        --model F:/Cabinet-BNN/results/deep_hash_mvp1_h512_s42.bin \
        --override F:/Cabinet-BNN/results/sim_override_mvp1_h512_s42.bin \
        --tag repro3109 --no-train --no-post

    # 新词表：训练 + 后处理
    python scripts/52_eval_pipeline.py --vocab data/natural_6000.txt \
        --generate-embeddings --tag nat6000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.data.embedding_cache import (  # noqa: E402
    normalize_rows,
    read_cache,
    write_cache,
)
from cabinet_bnn.hsh import codec  # noqa: E402
from cabinet_bnn.hsh.deep_hash import (  # noqa: E402
    bits_to_u64,
    export_deep_hash,
    project,
    quantize_bits,
    train_deep_hash,
)
from cabinet_bnn.hsh.metrics import evaluate_codes, recall_curve_paper  # noqa: E402
from cabinet_bnn.hsh.post_optimize import (  # noqa: E402
    export_override_cache,
    greedy_flip_optimize,
    hamming_matrix,
    load_override_cache,
    recall_at_k_from_ranking,
)


# ---------------------------------------------------------------- 嵌入

def generate_teacher_embeddings(words: list[str], model_path: Path,
                                out: Path, batch_size: int = 64) -> Path:
    """用本地 bge-large 生成教师嵌入缓存。"""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [嵌入] 加载 {model_path}  device={device}")
    m = SentenceTransformer(str(model_path), device=device)
    dim = m.get_sentence_embedding_dimension()
    print(f"  [嵌入] 编码 {len(words):,} 词（{dim} 维）…")
    t0 = time.time()
    vecs = m.encode(words, batch_size=batch_size, show_progress_bar=True,
                    normalize_embeddings=True, convert_to_numpy=True)
    print(f"  [嵌入] 完成 {time.time()-t0:.1f}s")
    write_cache(out, dim, zip(words, vecs.astype(np.float32)))
    print(f"  [嵌入] 写出 {out} ({out.stat().st_size/1024**2:.2f} MB)")
    return out


def make_student(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA 截断到 dim 维。返回 (student, pca_mean, pca_components)。"""
    if dim <= 0 or dim >= x.shape[1]:
        return x, np.zeros(x.shape[1], np.float32), np.eye(x.shape[1], dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    # 用 SVD 求主成分
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:dim].astype(np.float32)
    proj = normalize_rows((xc @ comps.T).astype(np.float32))
    return proj, mean.ravel(), comps


def load_or_make_pca(x: np.ndarray, dim: int, basis_path: Path | None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """若有已存的 PCA 基则复用（保证跨词表一致），否则重算。"""
    if basis_path and basis_path.exists():
        d = np.load(basis_path)
        mean, comps = d["mean"], d["components"]
        if comps.shape[0] != dim:
            raise ValueError(f"PCA 基维度 {comps.shape[0]} != 请求 {dim}")
        proj = normalize_rows(((x - mean) @ comps.T).astype(np.float32))
        print(f"  [PCA] 复用 {basis_path}  ({comps.shape[0]} 维)")
        return proj, mean, comps
    proj, mean, comps = make_student(x, dim)
    print(f"  [PCA] 新算：1024 -> {proj.shape[1]} 维")
    return proj, mean, comps


# ---------------------------------------------------------------- 评测

def recall_table(bits: np.ndarray, teacher_sim: np.ndarray,
                 ks=(1, 5, 10, 20)) -> dict:
    """两种 Recall 口径，都返回。

    ⚠️ 踩过的坑：把「全排名」当 pos_indices 传进 recall_at_k_from_ranking，
    会让 |P_i| = N-1，而 retrieved 只有 K 个 —— 结果恒等于 K/(N-1)，
    看起来像「召回 0.003」这种荒谬值。

    口径一 `overlap@{k}`（论文口径，recall_curve_paper）：
        真值集 = teacher top-k，检索集 = Hamming top-k。
        注意：两边同时随 k 增大，故【不保证单调】。
        论文的 0.7404 就是 overlap@10。

    口径二 `fixed10@{k}`（标准 IR 口径）：
        真值集固定为 teacher top-10，检索集 = Hamming top-k。
        随 k 单调不减，且【跨词表规模可比】—— 规模曲线用这个。
    """
    out = {"overlap": {}, "fixed10": {}}
    n = bits.shape[0]
    ham = hamming_matrix(bits)

    # 口径一：论文 overlap@k
    for k in ks:
        out["overlap"][k] = float(
            recall_curve_paper(bits, teacher_sim, (k,))[k])

    # 口径二：真值集固定 top-10
    k_truth = min(10, n - 1)
    pos10 = np.argsort(-teacher_sim, axis=1)[:, 1:k_truth + 1]
    for k in ks:
        kk = min(k, n - 1)
        out["fixed10"][k] = float(
            recall_at_k_from_ranking(pos10, ham, kk))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="HSH-64 统一评估流水线")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, default=None,
                    help="教师嵌入缓存（1024 维）。不存在且给了 --generate-embeddings 则生成")
    ap.add_argument("--generate-embeddings", action="store_true")
    ap.add_argument("--embed-model", type=Path,
                    default=Path(r"F:\hsh64_models\bge-large-zh-v1.5"))
    ap.add_argument("--student-dim", type=int, default=512)
    ap.add_argument("--pca-basis", type=Path, default=None,
                    help="PCA 基 .npz（含 mean/components）。不存在则新算并保存")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-bits", type=int, default=52)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--post-iters", type=int, default=10)
    ap.add_argument("--neg-weight", type=float, default=1.0)
    ap.add_argument("--pos-k", type=int, default=10)
    ap.add_argument("--model", type=Path, default=None, help="已有 DeepHash 模型（跳过训练）")
    ap.add_argument("--override", type=Path, default=None, help="已有 sim_override（跳过后处理）")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/eval")
    ap.add_argument("--jsonl", type=Path, default=ROOT / "results/eval_runs.jsonl")
    args = ap.parse_args()

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"HSH-64 评估流水线  tag={args.tag}")
    print("=" * 76)

    # ---- 1. 词表 ----
    words = [l.strip() for l in
             args.vocab.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(words)
    print(f"\n  [词表] {args.vocab.name}  {n:,} 词")

    # ---- 2. 教师嵌入 ----
    tc = args.teacher_cache or (args.out_dir / f"teacher_{args.tag}.cache")
    if not tc.exists():
        if not args.generate_embeddings:
            print(f"  ❌ 教师缓存不存在且未给 --generate-embeddings: {tc}")
            return 1
        tc = generate_teacher_embeddings(words, args.embed_model, tc)
    dim_t, tw, teacher = read_cache(tc)
    if tw != words:
        # 词表顺序不一致的常见原因：缓存对应别的词表
        if len(tw) != n:
            print(f"  ⚠️ 缓存词数 {len(tw)} != 词表 {n}，按词表重建索引")
            idx = {w: i for i, w in enumerate(tw)}
            miss = [w for w in words if w not in idx]
            if miss:
                print(f"  ❌ 缓存缺少 {len(miss)} 个词，例如 {miss[:5]}")
                return 1
            teacher = teacher[[idx[w] for w in words]]
    print(f"  [教师] {dim_t} 维  {teacher.shape}")

    # ---- 3. 学生输入（PCA）----
    basis_path = args.pca_basis or (args.out_dir / f"pca{args.student_dim}_{args.tag}.npz")
    student, pmean, pcomps = load_or_make_pca(teacher, args.student_dim, basis_path)
    np.savez(basis_path, mean=pmean, components=pcomps)
    print(f"  [学生] {student.shape}  基 -> {basis_path}")

    # 存成 Rust 能读的 cache
    student_cache = args.out_dir / f"student_{args.tag}_d{args.student_dim}.cache"
    write_cache(student_cache, student.shape[1], zip(words, student))
    print(f"  [学生] 缓存 -> {student_cache}  "
          f"({student_cache.stat().st_size/1024**2:.2f} MB)")

    # ---- 4. 教师相似度（真值）----
    tn = normalize_rows(teacher)
    teacher_sim = tn @ tn.T
    print(f"  [真值] 教师余弦矩阵 {teacher_sim.shape}")

    # ---- 5. 训练 ----
    model_path = args.model
    if args.no_train and model_path is None:
        print("  ❌ --no-train 但没给 --model")
        return 1
    if model_path is None:
        print(f"\n  [训练] {student.shape[1]} -> {args.hidden_dim} -> {args.n_bits}"
              f"  epochs={args.epochs}")
        t0 = time.time()
        mean, model, hist = train_deep_hash(
            student, n_bits=args.n_bits, hidden_dim=args.hidden_dim,
            epochs=args.epochs, lr=args.lr, pos_k=args.pos_k,
            teacher_x=teacher, seed=args.seed, device="cuda", log_every=100,
        )
        print(f"  [训练] 完成 {time.time()-t0:.1f}s")
        model_path = args.out_dir / f"deep_hash_{args.tag}.bin"
        export_deep_hash(model_path, mean, model)
        print(f"  [训练] 模型 -> {model_path} ({model_path.stat().st_size:,} bytes)")
    else:
        print(f"\n  [训练] 跳过，用 {model_path}")

    # ---- 6. 编码 ----
    print("  [编码] 载入模型并投影 …")
    mean_v, model_v = _load_exported(model_path)
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model_v = model_v.to(dev)
    u = project(model_v, mean_v, student, device=dev)
    sim_bits = quantize_bits(u)
    sim_vals = bits_to_u64(sim_bits)
    print(f"  [编码] 唯一 sim 码 {len(np.unique(sim_vals)):,} / {n:,} "
          f"= {len(np.unique(sim_vals))/n:.2%}")

    # ---- 7. feat + abs ----
    from cabinet_bnn.hsh.pos_map import assign_feat
    feats, fstats = assign_feat(words, "noun")   # 复现论文的硬编码行为
    feat_arr = np.array(feats, dtype=np.uint8)

    # ---- 8. 后处理前 ----
    print("\n" + "-" * 76)
    print("【后处理前】")
    rec_pre = recall_table(sim_bits, teacher_sim)
    _print_recall(rec_pre)

    # ---- 9. 后处理 ----
    bits_opt = sim_bits
    override_path = args.override
    if not args.no_post and override_path is None:
        print(f"\n  [后处理] 贪心比特翻转 iters={args.post_iters} "
              f"neg_weight={args.neg_weight}")
        pos_idx = np.argsort(-teacher_sim, axis=1)[:, 1:args.pos_k + 1]
        rec0 = recall_at_k_from_ranking(pos_idx, hamming_matrix(sim_bits), 10)
        print(f"    init Recall@10 = {rec0:.4f}")
        t0 = time.time()
        bits_opt = greedy_flip_optimize(
            sim_bits, teacher_sim, pos_k=args.pos_k, k=10,
            neg_weight=args.neg_weight, max_iters=args.post_iters, verbose=True,
        )
        print(f"    用时 {time.time()-t0:.1f}s")
        override_path = args.out_dir / f"sim_override_{args.tag}.bin"
        export_override_cache(override_path, words, bits_opt)
        print(f"    -> {override_path}")
    elif override_path is not None:
        print(f"\n  [后处理] 跳过，用 {override_path}")
        ow, obits = load_override_cache(override_path)
        idx = {w: i for i, w in enumerate(ow)}
        missing = [w for w in words if w not in idx]
        if missing:
            print(f"    ⚠️ override 缺少 {len(missing)} 词，例如 {missing[:5]}")
        bits_opt = np.zeros_like(sim_bits)
        for i, w in enumerate(words):
            if w in idx:
                bits_opt[i] = obits[idx[w]]
        print(f"    已载入 {len(ow)} 条")

    # ---- 10. 后处理后 ----
    rec_post = recall_table(bits_opt, teacher_sim)
    print("\n【后处理后】")
    _print_recall(rec_post)

    # ---- 11. 汇总落盘 ----
    rec = {
        "tag": args.tag,
        "n_words": n,
        "vocab": str(args.vocab),
        "student_dim": int(student.shape[1]),
        "hidden_dim": args.hidden_dim,
        "n_bits": args.n_bits,
        "seed": args.seed,
        "epochs": 0 if args.no_train else args.epochs,
        "unique_sim": int(len(np.unique(sim_vals))),
        "recall_pre": {s: {str(k): float(v) for k, v in d.items()}
                       for s, d in rec_pre.items()},
        "recall_post": {s: {str(k): float(v) for k, v in d.items()}
                        for s, d in rec_post.items()},
        "model": str(model_path),
        "override": str(override_path) if override_path else None,
        "student_cache": str(student_cache),
        "pca_basis": str(basis_path),
        "seconds": round(time.time() - t_start, 1),
    }
    run_json = args.out_dir / f"eval_{args.tag}.json"
    run_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  结果 -> {run_json}")
    print(f"  汇总 -> {args.jsonl}")
    return 0


def _print_recall(rec: dict) -> None:
    print(f"    {'K':>4}{'overlap@K(论文)':>18}{'fixed10@K(标准)':>18}")
    for k in rec["overlap"]:
        print(f"    {k:>4}{rec['overlap'][k]:>18.4f}"
              f"{rec['fixed10'][k]:>18.4f}")


def _load_exported(path: Path):
    """读 export_deep_hash 写出的二进制（与 Rust 同格式）。"""
    import struct

    import torch

    from cabinet_bnn.hsh.deep_hash import DeepHashMLP

    data = path.read_bytes()
    magic, ver, dim, hidden, n_bits = struct.unpack(">5I", data[:20])
    off = 20

    def f32(cnt):
        nonlocal off
        a = np.frombuffer(data[off:off + cnt * 4], dtype=">f4").copy().astype(np.float32)
        off += cnt * 4
        return a

    mean = f32(dim)
    W1 = f32(hidden * dim).reshape(hidden, dim)
    b1 = f32(hidden)
    gamma = f32(hidden)
    beta = f32(hidden)
    rm = f32(hidden)
    rv = f32(hidden)
    W2 = f32(n_bits * hidden).reshape(n_bits, hidden)
    b2 = f32(n_bits)

    model = DeepHashMLP(dim, hidden, n_bits)
    model.load_state_dict({
        "fc1.weight": torch.from_numpy(W1), "fc1.bias": torch.from_numpy(b1),
        "bn1.weight": torch.from_numpy(gamma), "bn1.bias": torch.from_numpy(beta),
        "bn1.running_mean": torch.from_numpy(rm),
        "bn1.running_var": torch.from_numpy(rv),
        "fc2.weight": torch.from_numpy(W2), "fc2.bias": torch.from_numpy(b2),
    })
    model.eval()
    return mean, model


if __name__ == "__main__":
    raise SystemExit(main())
