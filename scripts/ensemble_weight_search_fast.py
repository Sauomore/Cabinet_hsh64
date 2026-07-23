#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速 Ensemble 权重搜索（基于 300 query 验证集）。"""
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


def evaluate(D_list, weights, S_gt, top_k=10, pool_k=200, val_queries=None):
    N = S_gt.shape[0]
    if val_queries is None:
        val_queries = np.arange(N)
    recalls = []
    for i in val_queries:
        gt_pos = set(np.argsort(-S_gt[i])[1:top_k + 1])
        seen = set()
        cand_scores = {}
        cand_freq = {}
        for idx, D in enumerate(D_list):
            order = np.argsort(D[i])
            order = order[order != i]
            for rank, j in enumerate(order[:pool_k]):
                if j not in seen:
                    seen.add(j)
                    cand_scores[j] = 0.0
                    cand_freq[j] = 0
                cand_scores[j] += weights[idx] * D[i, j]
                cand_freq[j] += 1
        # 按 score 排序（score 越小越好）
        items = [(j, s) for j, s in cand_scores.items()]
        items.sort(key=lambda x: x[1])
        retrieved = [j for j, _ in items[:top_k]]
        hits = len(gt_pos & set(retrieved))
        recalls.append(hits / top_k)
    return np.mean(recalls)


def main():
    emb_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])
    override_paths = [Path(p) for p in sys.argv[3:]]

    _, emb_words, emb_X = parse_embedding_cache(emb_path)
    _, gt_words, gt_X = parse_embedding_cache(gt_path)
    if emb_words != gt_words:
        emb_word2idx = {w: i for i, w in enumerate(emb_words)}
        emb_order = [emb_word2idx[w] for w in gt_words]
        emb_X = emb_X[emb_order]
        emb_words = [emb_words[i] for i in emb_order]

    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)
    S_gt = gt_X_norm @ gt_X_norm.T
    N = len(emb_words)

    code_sets = []
    for ov_path in override_paths:
        ov_words, codes = parse_override(ov_path)
        word2idx = {w: i for i, w in enumerate(ov_words)}
        order = [word2idx[w] for w in emb_words]
        code_sets.append(codes[order])

    D_list = []
    for codes in code_sets:
        D = (codes[:, None, :] != codes[None, :, :]).sum(axis=2).astype(np.float32)
        np.fill_diagonal(D, 9999)
        D_list.append(D)

    # 固定验证集
    np.random.seed(2024)
    val_queries = np.random.choice(N, size=min(300, N), replace=False)

    best = (0.0, None, None)
    n = len(D_list)
    weight_options = []
    for combo in itertools.product([0.5, 1.0, 1.5, 2.0], repeat=n):
        if all(w == 0.5 for w in combo):
            continue
        weight_options.append(combo)

    pool_ks = [50, 100, 200]

    for weights in weight_options:
        for pool_k in pool_ks:
            rec = evaluate(D_list, weights, S_gt, top_k=10, pool_k=pool_k, val_queries=val_queries)
            if rec > best[0]:
                best = (rec, weights, pool_k)

    print(f"验证集最佳 Recall@10: {best[0]:.4f}")
    print(f"  权重: {best[1]}")
    print(f"  pool_k: {best[2]}")

    # 全量测试最佳配置
    full_rec = evaluate(D_list, best[1], S_gt, top_k=10, pool_k=best[2])
    print(f"全量 Recall@10: {full_rec:.4f}")

    print("\n单模型 Recall@10:")
    for idx, ov_path in enumerate(override_paths):
        rec = evaluate([D_list[idx]], [1.0], S_gt, top_k=10, pool_k=200)
        print(f"  {ov_path.name}: {rec:.4f}")


if __name__ == "__main__":
    main()
