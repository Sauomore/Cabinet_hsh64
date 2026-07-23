#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble 策略搜索 v2：支持按候选频次、平均排名、最小距离融合。"""
from __future__ import annotations

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


def evaluate(D_list, S_gt, top_k=10, pool_k=200, strategy="freq", weights=None):
    N = S_gt.shape[0]
    recalls = []
    for i in range(N):
        gt_pos = set(np.argsort(-S_gt[i])[1:top_k + 1])
        cand_rank_sum = {}
        cand_freq = {}
        cand_min_dist = {}
        for idx, D in enumerate(D_list):
            order = np.argsort(D[i])
            order = order[order != i]
            for rank, j in enumerate(order[:pool_k]):
                cand_freq[j] = cand_freq.get(j, 0) + 1
                cand_rank_sum[j] = cand_rank_sum.get(j, 0) + (rank + 1)
                if j not in cand_min_dist or D[i, j] < cand_min_dist[j]:
                    cand_min_dist[j] = D[i, j]

        items = []
        for j in cand_freq:
            freq = cand_freq[j]
            avg_rank = cand_rank_sum[j] / freq
            min_dist = cand_min_dist[j]
            if strategy == "freq":
                score = (-freq, avg_rank, min_dist)
            elif strategy == "avg_rank":
                score = (avg_rank, -freq, min_dist)
            elif strategy == "min_dist":
                score = (min_dist, -freq, avg_rank)
            elif strategy == "weighted_dist" and weights is not None:
                # 加权距离，需要原始 D；这里简化为用 min_dist
                score = (min_dist, -freq, avg_rank)
            else:
                score = (-freq, avg_rank, min_dist)
            items.append((j, score))

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

    best = (0.0, None, None)
    for pool_k in [50, 100, 200, 500]:
        for strategy in ["freq", "avg_rank", "min_dist"]:
            rec = evaluate(D_list, S_gt, top_k=10, pool_k=pool_k, strategy=strategy)
            if rec > best[0]:
                best = (rec, pool_k, strategy)

    print(f"最佳 Recall@10: {best[0]:.4f}")
    print(f"  pool_k={best[1]}, strategy={best[2]}")

    print("\n单模型 Recall@10:")
    for idx, ov_path in enumerate(override_paths):
        rec = evaluate([D_list[idx]], S_gt, top_k=10, pool_k=200, strategy="avg_rank")
        print(f"  {ov_path.name}: {rec:.4f}")


if __name__ == "__main__":
    main()
