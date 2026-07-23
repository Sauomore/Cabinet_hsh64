#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSH-64 第三阶段后处理：对 52-bit sim 码做贪心翻转变优。

思路：
1. 用已有模型（PCA 或 Deep Hash）生成初始 52-bit sim 码；
2. 定义目标权重矩阵 W，正样本（top-K 余弦邻居）权重为正，负样本为负；
3. 对每个词的每一位，计算翻转后目标损失的变化 delta；
4. 只执行能降低损失的翻转；
5. 迭代直到收敛或达到最大轮数；
6. 导出 word -> sim_code 覆盖映射，供 Rust Encoder 使用。

输出二进制格式：
    [u32: magic = 0xCAB1_0PT1]
    [u32: version = 1]
    [u32: n_bits = 52]
    [u32: count]
    重复 count 次：
        [u16: word_len]
        [u8 × word_len: word utf-8]
        [u64: sim_code]
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
    """计算余弦相似度矩阵。"""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X_norm @ X_norm.T


def build_weight_matrix(S: np.ndarray, pos_k: int, neg_k: int, neg_weight: float) -> np.ndarray:
    """构建成对权重矩阵。

    对每个 i：
    - top-K 最相似的 j 赋予 +1 权重；
    - bottom-K（最难负样本）赋予 -neg_weight；
    - 其余为 0。
    """
    N = S.shape[0]
    W = np.zeros((N, N), dtype=np.float32)

    # 排除自身
    S_diag = S.copy()
    np.fill_diagonal(S_diag, -1e9)

    pos_indices = np.argsort(-S_diag, axis=1)[:, :pos_k]
    neg_indices = np.argsort(S_diag, axis=1)[:, :neg_k]

    for i in range(N):
        W[i, pos_indices[i]] = 1.0
        W[i, neg_indices[i]] = -neg_weight

    # 对称化：W[i][j] 和 W[j][i] 都非零时取平均，否则保留单边
    W = (W + W.T) / 2.0
    return W


def hamming_distance_matrix(C: np.ndarray) -> np.ndarray:
    """C: (N, n_bits) 0/1 矩阵，返回 Hamming 距离矩阵。"""
    # (N, 1, n_bits) xor (1, N, n_bits) -> sum
    return ((C[:, None, :] ^ C[None, :, :]).sum(axis=2)).astype(np.float32)


def objective(C: np.ndarray, W: np.ndarray) -> float:
    """目标函数：sum_{i<j} W[i][j] * Hamming(i, j)。"""
    D = hamming_distance_matrix(C)
    # 只取上三角，避免重复计数
    return float((W * np.triu(D, k=1)).sum())


def greedy_flip_optimize(
    C: np.ndarray,
    W: np.ndarray,
    max_iters: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """贪心逐位翻转优化。"""
    N, n_bits = C.shape
    C = C.copy()

    # 预计算每个 bit 上每个词与其他词的匹配情况：match[i,j,b] = 1 if C[i,b] == C[j,b]
    # 翻转后距离变化：若当前相同，翻转后距离 +1；若不同，翻转后距离 -1
    # delta = sum_j W[i][j] * (1 - 2 * match[i,j,b])
    for it in range(max_iters):
        # (N, N, n_bits)
        match = 1 - (C[:, None, :] ^ C[None, :, :])
        # delta[i, b] = sum_j W[i, j] * (1 - 2 * match[i, j, b])
        delta = (W[:, :, None] * (1 - 2 * match)).sum(axis=1)

        # 只取负 delta（能降低目标）
        # 但每轮只翻转最有益的一位，避免冲突；也可以批量翻转无冲突位
        # 这里采用：找出所有 delta < 0 的位置，然后按 delta 排序依次应用，
        # 应用后重新计算该行的 delta（因为会影响与其他词的匹配）
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
            # 执行翻转
            C[i, b] ^= 1
            improved = True
            flipped_rows.add(i)

        loss = objective(C, W)
        if verbose:
            print(f"[iter {it + 1}] flipped={len(flipped_rows)} loss={loss:.2f}", file=sys.stderr)
        if not improved:
            break

    return C


def export_override_cache(path: Path, words: list[str], codes: np.ndarray) -> None:
    """导出 word -> sim_code 覆盖缓存（与 Rust 约定一致：bit i -> 整数 bit i）。"""
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
        # bitorder="little" 让 code[0] 位于字节 0 的 bit 0，与 Rust `sim |= 1 << i` 一致
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
    """加载 embedding 并生成初始 52-bit 二进制码。

    返回 (words, X, C, sim_X)，其中 sim_X 用于构建优化目标（默认与 X 相同，
    蒸馏场景下可由教师向量提供）。
    """
    dim, words, X = parse_embedding_cache(embedding_cache)
    N = X.shape[0]

    sim_X = parse_embedding_cache(teacher_cache)[2] if teacher_cache is not None else X

    if model_type == "pca":
        from train_pca_64 import load_pca_cache
        mean, components, _ = load_pca_cache(model_path)
        Xc = X - mean
        proj = Xc @ components.T  # (N, 52)
        C = (proj >= 0).astype(np.uint8)
    elif model_type == "deep_hash":
        if model_path is not None:
            print(f"[load] Deep Hash 模型: {model_path}", file=sys.stderr)
            mean, model = train_deep_hash_v3_64.load_deep_hash_model(model_path)
        else:
            print("[train] 重新训练 Deep Hash 模型", file=sys.stderr)
            mean, model, _ = train_deep_hash_v3_64.train_deep_hash_v3(
                X,
                n_bits=52,
                hidden_dim=512,
                epochs=500,
            )
        X_tensor = train_deep_hash_v3_64.torch.from_numpy((X - mean).astype(np.float32))
        model.eval()
        with train_deep_hash_v3_64.torch.no_grad():
            u = model(X_tensor)
            C = (u >= 0).cpu().numpy().astype(np.uint8)
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    assert C.shape == (N, 52), f"码矩阵形状错误: {C.shape}"
    return words, X, C, sim_X


def main() -> int:
    parser = argparse.ArgumentParser(description="HSH-64 sim 码后处理贪心翻转变优")
    parser.add_argument("--embedding-cache", type=Path, required=True, help="embedding 缓存路径")
    parser.add_argument("--model-type", choices=["pca", "deep_hash"], default="pca", help="初始码来源")
    parser.add_argument("--model-path", type=Path, help="PCA 或 Deep Hash 模型路径（deep_hash 可省略，将重新训练）")
    parser.add_argument("-o", "--output", type=Path, default=Path("sim_override_64.bin"), help="输出覆盖缓存路径")
    parser.add_argument("--pos-k", type=int, default=10, help="正样本数")
    parser.add_argument("--neg-k", type=int, default=50, help="负样本数")
    parser.add_argument("--neg-weight", type=float, default=0.2, help="负样本权重")
    parser.add_argument("--max-iters", type=int, default=10, help="最大迭代轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--teacher-cache", type=Path, default=None, help="教师向量缓存（蒸馏场景下优化目标）")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.model_type == "deep_hash" and args.model_path is None:
        print("[warn] 未提供 Deep Hash 模型路径，将重新训练（耗时）", file=sys.stderr)

    words, X, C, sim_X = load_initial_codes(
        args.embedding_cache, args.model_path, args.model_type, args.teacher_cache
    )
    unique_init = len({bytes(r) for r in np.packbits(C, axis=1, bitorder="little")})
    print(f"[init] {len(words)} 个词，初始唯一码 {unique_init}", file=sys.stderr)

    S = cosine_similarity_matrix(sim_X)
    W = build_weight_matrix(S, args.pos_k, args.neg_k, args.neg_weight)
    print(f"[obj] 初始目标值: {objective(C, W):.2f}", file=sys.stderr)

    C_opt = greedy_flip_optimize(C, W, args.max_iters)
    unique_opt = len({bytes(r) for r in np.packbits(C_opt, axis=1, bitorder="little")})
    print(f"[obj] 优化后目标值: {objective(C_opt, W):.2f}", file=sys.stderr)
    print(f"[unique] 优化后唯一码 {unique_opt}", file=sys.stderr)

    export_override_cache(args.output, words, C_opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
