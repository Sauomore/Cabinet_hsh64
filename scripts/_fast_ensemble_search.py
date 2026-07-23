#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速 Ensemble 组合搜索：先单模型排序，再贪心扩展。"""
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


def evaluate(D_list, S_gt, top_k=10, pool_k=200, strategy="freq"):
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
            else:
                score = (-freq, avg_rank, min_dist)
            items.append((j, score))

        items.sort(key=lambda x: x[1])
        retrieved = [j for j, _ in items[:top_k]]
        hits = len(gt_pos & set(retrieved))
        recalls.append(hits / top_k)
    return np.mean(recalls)


def main():
    emb_path = Path("tests/data/embedding_3109.cache")
    gt_path = Path("tests/data/reranker_3109_large_aligned.cache")

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

    # 加载所有兼容的 override
    override_files = sorted(Path("tests/data").glob("sim_override_*.bin"))
    candidates = []
    for ov_path in override_files:
        try:
            ov_words, codes = parse_override(ov_path)
        except Exception:
            continue
        if ov_words != emb_words:
            continue
        D = (codes[:, None, :] != codes[None, :, :]).sum(axis=2).astype(np.float32)
        np.fill_diagonal(D, 9999)
        candidates.append((ov_path.name, D))

    print(f"加载兼容模型数: {len(candidates)}", file=sys.stderr)

    # 单模型评估
    single_scores = []
    for name, D in candidates:
        rec = evaluate([D], S_gt, top_k=10, pool_k=200, strategy="avg_rank")
        single_scores.append((rec, name, D))
    single_scores.sort(reverse=True)

    print("\nTop 20 单模型:", file=sys.stderr)
    for rec, name, _ in single_scores[:20]:
        print(f"  {rec:.4f}  {name}", file=sys.stderr)

    # 贪心前向选择：从空集开始，每次加入能提升最多的模型
    top_models = single_scores[:25]
    best_ensembles = []

    for strategy in ["freq", "avg_rank", "min_dist"]:
        selected = []
        selected_D = []
        remaining = list(top_models)
        while remaining and len(selected) < 8:
            best = (-1.0, None)
            for rec, name, D in remaining:
                trial_D = selected_D + [D]
                for pool_k in [50, 100, 200, 500, 1000, 2000]:
                    rec_e = evaluate(trial_D, S_gt, top_k=10, pool_k=pool_k, strategy=strategy)
                    if rec_e > best[0]:
                        best = (rec_e, (name, pool_k))
            if best[0] <= evaluate(selected_D, S_gt, top_k=10, pool_k=200, strategy=strategy) if selected_D else 0.0:
                # 没有提升就停止
                if not selected_D:
                    selected_D.append(remaining[0][2])
                    selected.append(remaining[0][1])
                break
            # 找到最佳加入模型
            best_name, best_pool = best[1]
            for idx, (rec, name, D) in enumerate(remaining):
                if name == best_name:
                    selected.append(name)
                    selected_D.append(D)
                    remaining.pop(idx)
                    break
            rec_final = evaluate(selected_D, S_gt, top_k=10, pool_k=best_pool, strategy=strategy)
            best_ensembles.append((rec_final, strategy, best_pool, list(selected)))
            print(f"[{strategy}] size={len(selected)} rec={rec_final:.4f} pool_k={best_pool} {selected}", file=sys.stderr)

    best_ensembles.sort(reverse=True)
    print("\n全局 Top 20 Ensemble:")
    for rec, strategy, pool_k, names in best_ensembles[:20]:
        print(f"  {rec:.4f}  strategy={strategy} pool_k={pool_k}  models={names}")


if __name__ == "__main__":
    main()
