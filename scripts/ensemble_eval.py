#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 HSH Ensemble 评测：多模型候选并集 + Hamming 排序，无 embedding 精排。"""
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


def main():
    emb_path = Path(sys.argv[1])
    gt_path = Path(sys.argv[2])
    override_paths = [Path(p) for p in sys.argv[3:]]
    top_k = 10

    _, emb_words, emb_X = parse_embedding_cache(emb_path)
    _, gt_words, gt_X = parse_embedding_cache(gt_path)
    assert emb_words == gt_words

    # 教师相似度
    gt_X_norm = gt_X / (np.linalg.norm(gt_X, axis=1, keepdims=True) + 1e-8)
    S = gt_X_norm @ gt_X_norm.T

    # 加载所有 override 码并对齐到 emb_words 顺序
    code_sets = []
    for ov_path in override_paths:
        ov_words, codes = parse_override(ov_path)
        word2idx = {w: i for i, w in enumerate(ov_words)}
        order = [word2idx[w] for w in emb_words]
        code_sets.append(codes[order])

    N = len(emb_words)
    recalls = []
    qps_like = []
    for i in range(N):
        gt_pos = set(np.argsort(-S[i])[1:top_k + 1])
        # 并集：取所有模型中 Hamming 距离前 top_k*20 的候选
        seen = set()
        candidates = []
        for codes in code_sets:
            D = (codes[i] != codes).sum(axis=1)
            D[i] = 9999
            order = np.argsort(D)
            for j in order[:top_k * 20]:
                if j not in seen:
                    seen.add(j)
                    candidates.append((j, D[j]))
        candidates.sort(key=lambda x: x[1])
        retrieved = [j for j, _ in candidates[:top_k]]
        hits = len(gt_pos & set(retrieved))
        recalls.append(hits / top_k)

    print(f"Ensemble 模型数: {len(code_sets)}")
    print(f"平均 Recall@{top_k}: {np.mean(recalls):.4f}")


if __name__ == "__main__":
    main()
