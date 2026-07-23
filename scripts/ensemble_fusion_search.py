#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble 融合策略搜索：多模型候选并集 + 加权/频次排序。

用法：
    python ensemble_fusion_search.py <embedding.cache> <gt.cache> <override1.bin> [<override2.bin> ...]
"""
from __future__ import annotations

import itertools
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_pca_64 import parse_embedding_cache


def parse_override(path: Path):
    with path.open("rb") as f:
        magic, ver, n_bits, n = struct.unpack(">IIII", f.read(16))
        words = []
        codes = np.zeros((n, n_bits), dtype=np.uint8)
        for i in range(n):
            l = struct.unpack(">H", f.read(2))[0]
            word = f.read(l).decode("utf-8")
            words.append(word)
            sim = struct.unpack(">Q", f.read(8))[0]
            bits = np.unpackbits(np.frombuffer(sim.to_bytes(8, "little"), dtype=np.uint8), bitorder="little")[:n_bits]
            codes[i] = bits
        return words, codes


def compute_recall(code_sets, weights, top_k, S_gt, candidate_pool_k, freq_weight=0.0, rank_weight=0.0):
    """给定模型权重，计算 Recall@top_k。

    - code_sets: list of (N, N) Hamming distance matrices
    - weights: list of float, 与 code_sets 一一对应
    - candidate_pool_k: 每个模型取前 candidate_pool_k 个候选构成并集
    - freq_weight: 候选被多少模型召回的奖励权重
    - rank_weight: 按每个模型内部排名的奖励权重
    """
    N = S_gt.shape[0]
    recalls = []
    for i in range(N):
        gt_pos = set(np.argsort(-S_gt[i])[1:top_k + 1])

        seen = set()
        candidates = []
        model_ranks = []
        for idx, D in enumerate(code_sets):
            order = np.argsort(D[i])
            order = order[order != i]
            for rank, j in enumerate(order[:candidate_pool_k]):
                if j not in seen:
                    seen.add(j)
                    candidates.append(j)
                    model_ranks.append({})
                model_ranks[candidates.index(j)][idx] = rank + 1

        scores = []
        for j, ranks in zip(candidates, model_ranks):
            score = 0.0
            for idx, D in enumerate(code_sets):
                score += weights[idx] * D[i, j]
            if freq_weight != 0.0:
                score -= freq_weight * len(ranks)
            if rank_weight != 0.0:
                avg_inv_rank = sum(1.0 / r for r in ranks.values()) / len(code_sets)
                score -= rank_weight * avg_inv_rank
            scores.append((j, score))

        scores.sort(key=lambda x: x[1])
        retrieved = [j for j, _ in scores[:top_k]]
        hits = len(gt_pos & set(retrieved))
        recalls.append(hits / top_k)

    return np.mean(recalls)


def main():
    emb_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])
    override_paths = [Path(p) for p in sys.argv[3:]]
    top_k = 10

    _, emb_words, emb_X = parse_embedding_cache(emb_path)
    _, gt_words, gt_X = parse_embedding_cache(gt_path)
    if emb_words != gt_words:
        emb_word2idx = {w: i for i, w in enumerate(emb_words)}
        emb_order = [emb_word2idx[w] for w in gt_words]
        emb_X = emb_X[emb_order]
        emb_words = [emb_words[i] for i in emb_order]

    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)
    S_gt = gt_X_norm @ gt_X_norm.T

    # 加载所有 override 并对齐
    code_sets_bits = []
    for ov_path in override_paths:
        ov_words, codes = parse_override(ov_path)
        word2idx = {w: i for i, w in enumerate(ov_words)}
        order = [word2idx[w] for w in emb_words]
        aligned = codes[order]
        code_sets_bits.append(aligned)

    # 预计算 Hamming 距离矩阵
    N = len(emb_words)
    dist_mats = []
    for codes in code_sets_bits:
        D = (codes[:, None, :] != codes[None, :, :]).sum(axis=2).astype(np.float32)
        np.fill_diagonal(D, 9999)
        dist_mats.append(D)

    best = (0.0, None, None, None)

    # 搜索空间
    candidate_pool_ks = [50, 100, 200]
    freq_weights = [0.0, 0.5, 1.0, 2.0]
    rank_weights = [0.0, 1.0, 5.0, 10.0]

    n_models = len(dist_mats)
    # 简单权重：等权、按 recall 加权、少数离散组合
    weight_options = []
    weight_options.append(tuple([1.0] * n_models))  # 等权
    weight_options.append(tuple([0.5, 1.0, 1.0, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 0.5, 1.0, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 1.0, 0.5, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 1.0, 1.0, 0.5][:n_models]))
    weight_options.append(tuple([2.0, 1.0, 1.0, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 2.0, 1.0, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 1.0, 2.0, 1.0][:n_models]))
    weight_options.append(tuple([1.0, 1.0, 1.0, 2.0][:n_models]))

    for weights in weight_options:
        for pool_k, fw, rw in itertools.product(candidate_pool_ks, freq_weights, rank_weights):
            rec = compute_recall(dist_mats, weights, top_k, S_gt, pool_k, fw, rw)
            if rec > best[0]:
                best = (rec, weights, pool_k, (fw, rw))

    print(f"最佳 Recall@{top_k}: {best[0]:.4f}")
    print(f"  权重: {best[1]}")
    print(f"  候选池: 每个模型前 {best[2]} 个")
    print(f"  freq_weight={best[3][0]}, rank_weight={best[3][1]}")

    # 也输出单模型 Recall
    print("\n单模型 Recall@10:")
    for idx, ov_path in enumerate(override_paths):
        rec = compute_recall([dist_mats[idx]], [1.0], top_k, S_gt, top_k * 20, 0.0, 0.0)
        print(f"  {ov_path.name}: {rec:.4f}")


if __name__ == "__main__":
    main()
