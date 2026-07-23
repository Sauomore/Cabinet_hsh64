#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接计算给定 sim_override 缓存的纯 Hamming Recall@10，与 Rust benchmark 对齐。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_pca_64 import parse_embedding_cache


def parse_override_cache(path: Path):
    with path.open("rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ver = struct.unpack(">I", f.read(4))[0]
        n_bits = struct.unpack(">I", f.read(4))[0]
        n = struct.unpack(">I", f.read(4))[0]
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


def main():
    emb_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])
    override_path = Path(sys.argv[3])

    _, emb_words, emb_X = parse_embedding_cache(emb_path)
    _, gt_words, gt_X = parse_embedding_cache(gt_path)
    if emb_words != gt_words:
        print(f"警告：词表顺序不一致，emb[0]={emb_words[0]}, gt[0]={gt_words[0]}", file=sys.stderr)
        print(f"词集合相同: {set(emb_words) == set(gt_words)}", file=sys.stderr)
        # 按 gt_words 顺序重排 emb_X
        emb_word2idx = {w: i for i, w in enumerate(emb_words)}
        emb_order = [emb_word2idx[w] for w in gt_words]
        emb_X = emb_X[emb_order]
        emb_words = [emb_words[i] for i in emb_order]

    ov_words, ov_codes = parse_override_cache(override_path)
    # 对齐顺序到 emb_words
    word2idx = {w: i for i, w in enumerate(ov_words)}
    order = [word2idx[w] for w in emb_words]
    codes = ov_codes[order]

    # 归一化
    emb_X_norm = emb_X / (np.linalg.norm(emb_X, axis=1, keepdims=True) + 1e-8)
    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)

    emb_sim = emb_X_norm @ emb_X_norm.T
    gt_sim = gt_X_norm @ gt_X_norm.T

    N = len(emb_words)
    top_k = 10

    def recall(S_gt, S_pred):
        total = 0.0
        for i in range(N):
            gt_pos = np.argsort(-S_gt[i])[1:top_k + 1]
            # Hamming 距离
            D = (codes[i] != codes).sum(axis=1)
            D[i] = 9999
            pred_pos = np.argsort(D)[:top_k]
            hits = len(set(gt_pos) & set(pred_pos))
            total += hits / top_k
        return total / N

    print(f"Recall@10 (ground truth = embedding): {recall(emb_sim, emb_sim):.4f}")
    print(f"Recall@10 (ground truth = reranker):  {recall(gt_sim, gt_sim):.4f}")


if __name__ == "__main__":
    main()
