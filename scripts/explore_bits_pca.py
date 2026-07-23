#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探索不同 sim 码长度对纯 Hamming Recall@10 的影响（基于 PCA + 后处理）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from train_pca_64 import parse_embedding_cache


def recall_at_k(C, S_gt, k=10):
    N = len(S_gt)
    recalls = []
    for i in range(N):
        gt_pos = set(np.argsort(-S_gt[i])[1:k + 1])
        D = (C[i] != C).sum(axis=1)
        D[i] = 9999
        pred_pos = np.argsort(D)[:k]
        hits = len(gt_pos & set(pred_pos))
        recalls.append(hits / k)
    return np.mean(recalls)


def main():
    emb_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])

    _, emb_words, emb_X = parse_embedding_cache(emb_path)
    _, gt_words, gt_X = parse_embedding_cache(gt_path)
    if emb_words != gt_words:
        emb_word2idx = {w: i for i, w in enumerate(emb_words)}
        emb_order = [emb_word2idx[w] for w in gt_words]
        emb_X = emb_X[emb_order]
        emb_words = [emb_words[i] for i in emb_order]

    # 使用 bge-large 作为输入和 GT
    X = gt_X
    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)
    S_gt = gt_X_norm @ gt_X_norm.T

    print("bits\tinit_recall\tbalanced_recall")
    for n_bits in [52, 64, 80, 96, 116, 128]:
        pca = PCA(n_components=n_bits)
        Xc = X - X.mean(axis=0)
        proj = pca.fit_transform(Xc)
        C = (proj >= 0).astype(np.uint8)
        init_rec = recall_at_k(C, S_gt, k=10)

        # 简单 bit 平衡：对每个 bit，如果 >50% 为 1，翻转该 bit
        C_bal = C.copy()
        for b in range(n_bits):
            if C_bal[:, b].mean() > 0.5:
                C_bal[:, b] ^= 1
        bal_rec = recall_at_k(C_bal, S_gt, k=10)

        print(f"{n_bits}\t{init_rec:.4f}\t\t{bal_rec:.4f}")


if __name__ == "__main__":
    main()
