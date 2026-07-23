#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSH-64 后处理 v2：更激进的 Recall@K 优化。

改进点：
1. 每轮允许同一行翻转多个 bit（只要 delta < 0）。
2. 可选模拟退火：以概率接受少量 delta > 0 的翻转，帮助跳出局部最优。
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_deep_hash_v3_64
from post_optimize_codes_recall_64 import (
    build_recall_weight_matrix,
    cosine_similarity_matrix,
    export_override_cache,
    hamming_distance_matrix,
    load_initial_codes,
    recall_at_k_from_ranking,
)


def objective(C: np.ndarray, W: np.ndarray) -> float:
    D = hamming_distance_matrix(C)
    return float((W * np.triu(D, k=1)).sum())


def greedy_flip_optimize_recall_v2(
    C: np.ndarray,
    S: np.ndarray,
    pos_k: int,
    k: int,
    pos_weight: float,
    neg_weight: float,
    max_iters: int = 10,
    sa_temperature: float = 0.0,
    sa_cooldown: float = 0.95,
    verbose: bool = True,
) -> np.ndarray:
    """面向 Recall@K 的贪心翻转 v2。"""
    N, n_bits = C.shape
    C = C.copy()
    rng = np.random.default_rng(42)

    pos_indices = np.argsort(-S, axis=1)[:, 1 : pos_k + 1]

    temperature = sa_temperature

    for it in range(max_iters):
        W = build_recall_weight_matrix(C, S, pos_k, k, pos_weight, neg_weight)

        match = 1 - (C[:, None, :] ^ C[None, :, :])
        delta = (W[:, :, None] * (1 - 2 * match)).sum(axis=1)

        improved = False
        order = np.argsort(delta.flatten())

        for idx in order:
            i = idx // n_bits
            b = idx % n_bits
            d = delta[i, b]

            if d < -1e-6:
                C[i, b] ^= 1
                improved = True
            elif temperature > 0:
                # 模拟退火接受准则
                accept_prob = math.exp(-d / temperature) if d > 0 else 1.0
                if rng.random() < accept_prob:
                    C[i, b] ^= 1
                    improved = True
            else:
                # 温度为零时跳过正 delta
                break

        temperature *= sa_cooldown

        loss = objective(C, W)
        rec = recall_at_k_from_ranking(pos_indices, hamming_distance_matrix(C), k)
        if verbose:
            print(
                f"[iter {it + 1}] loss={loss:.2f} recall@{k}={rec:.4f} temp={temperature:.4f}",
                file=sys.stderr,
            )
        if not improved and temperature <= 1e-6:
            break

    return C


def main() -> int:
    parser = argparse.ArgumentParser(description="HSH-64 sim 码 Recall@K 导向后处理 v2")
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--model-type", choices=["pca", "deep_hash"], default="deep_hash")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("sim_override_recall_64_v2.bin"))
    parser.add_argument("--pos-k", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--sa-temperature", type=float, default=0.0, help="模拟退火初始温度（0=禁用）")
    parser.add_argument("--sa-cooldown", type=float, default=0.95)
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

    C_opt = greedy_flip_optimize_recall_v2(
        C,
        S,
        args.pos_k,
        args.k,
        args.pos_weight,
        args.neg_weight,
        args.max_iters,
        args.sa_temperature,
        args.sa_cooldown,
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
