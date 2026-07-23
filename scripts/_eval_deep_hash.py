#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速评估 Deep Hash 模型纯 Hamming Recall@10。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_deep_hash_v3_64
from train_pca_64 import parse_embedding_cache


def main():
    model_path = Path(sys.argv[1])
    input_cache = Path(sys.argv[2])
    gt_cache = Path(sys.argv[3])

    _, words, X = parse_embedding_cache(input_cache)
    _, gt_words, gt_X = parse_embedding_cache(gt_cache)
    if words != gt_words:
        w2i = {w: i for i, w in enumerate(words)}
        order = [w2i[w] for w in gt_words]
        X = X[order]
        words = [words[i] for i in order]

    mean, model = train_deep_hash_v3_64.load_deep_hash_model(model_path)
    model.eval()
    with train_deep_hash_v3_64.torch.no_grad():
        u = model(train_deep_hash_v3_64.torch.from_numpy((X - mean).astype(np.float32)))
        codes = (u >= 0).cpu().numpy().astype(np.uint8)

    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)
    S_gt = gt_X_norm @ gt_X_norm.T

    N = len(words)
    top_k = 10
    total = 0.0
    for i in range(N):
        gt_pos = set(np.argsort(-S_gt[i])[1:top_k + 1])
        D = (codes[i] != codes).sum(axis=1)
        D[i] = 9999
        pred_pos = set(np.argsort(D)[:top_k])
        total += len(gt_pos & pred_pos) / top_k
    print(f"Recall@10 (gt=reranker): {total / N:.4f}")

    # 也输出与 bge-small 的对比
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S_small = X_norm @ X_norm.T
    total2 = 0.0
    for i in range(N):
        gt_pos = set(np.argsort(-S_small[i])[1:top_k + 1])
        D = (codes[i] != codes).sum(axis=1)
        D[i] = 9999
        pred_pos = set(np.argsort(D)[:top_k])
        total2 += len(gt_pos & pred_pos) / top_k
    print(f"Recall@10 (gt=embedding): {total2 / N:.4f}")


if __name__ == "__main__":
    main()
