#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSH-64 后处理：直接以 Recall@K 为目标动态调整权重进行贪心翻转变优。

与标准后处理不同，本脚本每轮根据当前 Hamming 距离排序动态挖掘"难负样本"，
让正样本（教师 top-K 邻居）的 Hamming 距离尽可能小，并把挤进前 K 的负样本推出去。
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_deep_hash_v3_64
from train_pca_64 import parse_embedding_cache

MAGIC = 0xCAB1_0D01
VERSION = 1


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X_norm @ X_norm.T


def hamming_distance_matrix(C: np.ndarray) -> np.ndarray:
    return ((C[:, None, :] ^ C[None, :, :]).sum(axis=2)).astype(np.float32)


def recall_at_k_from_ranking(pos_indices: np.ndarray, D: np.ndarray, k: int) -> float:
    """给定正样本索引和 Hamming 距离矩阵，计算 Recall@K（平均）。"""
    N = D.shape[0]
    total = 0.0
    for i in range(N):
        pos = set(pos_indices[i])
        if not pos:
            continue
        # 按 Hamming 距离排序取前 k（排除自身）
        order = np.argsort(D[i])
        order = order[order != i]
        retrieved = set(order[:k])
        total += len(pos & retrieved) / len(pos)
    return total / N


def build_recall_weight_matrix(
    C: np.ndarray,
    S: np.ndarray,
    pos_k: int,
    k: int,
    pos_weight: float,
    neg_weight: float,
) -> np.ndarray:
    """构建面向 Recall@K 的动态权重矩阵。

    对每个 i：
    - P_i：教师相似度 top-K 正样本；
    - H_i：当前 Hamming 距离前 k 小但不在 P_i 中的"难负样本"；
    - 权重 W[i, P_i] = +pos_weight，W[i, H_i] = -neg_weight。
    """
    N = C.shape[0]
    W = np.zeros((N, N), dtype=np.float32)

    S_diag = S.copy()
    np.fill_diagonal(S_diag, -1e9)
    pos_indices = np.argsort(-S_diag, axis=1)[:, :pos_k]

    D = hamming_distance_matrix(C)
    np.fill_diagonal(D, 1e9)
    # 取 Hamming 前 k*2 作为潜在难负样本池，再过滤
    pool_k = max(k * 3, pos_k * 2)
    ham_order = np.argsort(D, axis=1)[:, :pool_k]

    for i in range(N):
        pos_set = set(pos_indices[i])
        W[i, pos_indices[i]] = pos_weight
        for j in ham_order[i]:
            if j == i or j in pos_set:
                continue
            W[i, j] = -neg_weight
            if (W[i, :] < 0).sum() >= k:
                break

    # 对称化
    W = (W + W.T) / 2.0
    return W


def objective(C: np.ndarray, W: np.ndarray) -> float:
    D = hamming_distance_matrix(C)
    return float((W * np.triu(D, k=1)).sum())


def greedy_flip_optimize_recall(
    C: np.ndarray,
    S: np.ndarray,
    pos_k: int,
    k: int,
    pos_weight: float,
    neg_weight: float,
    max_iters: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """面向 Recall@K 的贪心翻转。"""
    N, n_bits = C.shape
    C = C.copy()

    for it in range(max_iters):
        W = build_recall_weight_matrix(C, S, pos_k, k, pos_weight, neg_weight)

        match = 1 - (C[:, None, :] ^ C[None, :, :])
        delta = (W[:, :, None] * (1 - 2 * match)).sum(axis=1)

        improved = False
        order = np.argsort(delta.flatten())
        flipped_rows = set()
        for idx in order:
            i = idx // n_bits
            b = idx % n_bits
            if i in flipped_rows:
                continue
            if delta[i, b] >= -1e-6:
                break
            C[i, b] ^= 1
            improved = True
            flipped_rows.add(i)

        loss = objective(C, W)
        rec = recall_at_k_from_ranking(np.argsort(-S, axis=1)[:, 1 : pos_k + 1], hamming_distance_matrix(C), k)
        if verbose:
            print(
                f"[iter {it + 1}] flipped={len(flipped_rows)} loss={loss:.2f} recall@{k}={rec:.4f}",
                file=sys.stderr,
            )
        if not improved:
            break

    return C


def export_override_cache(path: Path, words: list[str], codes: np.ndarray) -> None:
    assert codes.shape[1] == 52
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    buf.extend(struct.pack(">I", MAGIC))
    buf.extend(struct.pack(">I", VERSION))
    buf.extend(struct.pack(">I", 52))
    buf.extend(struct.pack(">I", len(words)))
    for word, code in zip(words, codes):
        wb = word.encode("utf-8")
        buf.extend(struct.pack(">H", len(wb)))
        buf.extend(wb)
        packed = np.packbits(code, bitorder="little")
        sim = int.from_bytes(packed.tobytes()[:7], "little")
        buf.extend(struct.pack(">Q", sim))
    path.write_bytes(buf)
    print(f"[write] {path} ({len(buf)} bytes)", file=sys.stderr)


def load_initial_codes(
    embedding_cache: Path,
    model_path: Path | None,
    model_type: str,
    teacher_cache: Path | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    dim, words, X = parse_embedding_cache(embedding_cache)
    N = X.shape[0]

    sim_X = parse_embedding_cache(teacher_cache)[2] if teacher_cache is not None else X

    if model_type == "pca":
        from train_pca_64 import load_pca_cache
        mean, components, _ = load_pca_cache(model_path)
        Xc = X - mean
        proj = Xc @ components.T
        C = (proj >= 0).astype(np.uint8)
    elif model_type == "deep_hash":
        if model_path is not None:
            print(f"[load] Deep Hash 模型: {model_path}", file=sys.stderr)
            mean, model = train_deep_hash_v3_64.load_deep_hash_model(model_path)
        else:
            print("[train] 重新训练 Deep Hash 模型", file=sys.stderr)
            mean, model, _ = train_deep_hash_v3_64.train_deep_hash_v3(
                X, n_bits=52, hidden_dim=512, epochs=500
            )
        X_tensor = train_deep_hash_v3_64.torch.from_numpy((X - mean).astype(np.float32))
        model.eval()
        with train_deep_hash_v3_64.torch.no_grad():
            u = model(X_tensor)
            C = (u >= 0).cpu().numpy().astype(np.uint8)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    assert C.shape == (N, 52)
    return words, X, C, sim_X


def main() -> int:
    parser = argparse.ArgumentParser(description="HSH-64 sim 码 Recall@K 导向后处理")
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--model-type", choices=["pca", "deep_hash"], default="deep_hash")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("sim_override_recall_64.bin"))
    parser.add_argument("--pos-k", type=int, default=10, help="教师向量 top-K 正样本")
    parser.add_argument("--k", type=int, default=10, help="目标 Recall@K 的 K")
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neg-weight", type=float, default=2.0, help="难负样本权重")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-cache", type=Path, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    words, X, C, sim_X = load_initial_codes(
        args.embedding_cache, args.model_path, args.model_type, args.teacher_cache
    )
    S = cosine_similarity_matrix(sim_X)
    init_recall = recall_at_k_from_ranking(
        np.argsort(-S, axis=1)[:, 1 : args.pos_k + 1],
        hamming_distance_matrix(C),
        args.k,
    )
    print(f"[init] {len(words)} 个词，初始 Recall@{args.k}={init_recall:.4f}", file=sys.stderr)

    C_opt = greedy_flip_optimize_recall(
        C,
        S,
        args.pos_k,
        args.k,
        args.pos_weight,
        args.neg_weight,
        args.max_iters,
    )
    final_recall = recall_at_k_from_ranking(
        np.argsort(-S, axis=1)[:, 1 : args.pos_k + 1],
        hamming_distance_matrix(C_opt),
        args.k,
    )
    print(f"[final] Recall@{args.k}={final_recall:.4f}", file=sys.stderr)

    export_override_cache(args.output, words, C_opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
