#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端语义哈希训练 v3（HSH-64 版本）：MLP + STE + 多目标损失。

相比线性 PCA，v3 使用一个小型 MLP（dim -> hidden -> n_bits）
配合 Straight-Through Estimator，在保持 52-bit 二进制码的前提下，
更灵活地保持 embedding 空间的邻域结构。

输出二进制格式兼容 Rust DeepHashProjection：

    [u32: magic = 0xCAB1_DE3D]
    [u32: version = 1]
    [u32: dim]
    [u32: hidden_dim]
    [u32: n_bits]
    [f32 × dim: mean]
    [f32 × hidden_dim × dim: W1]
    [f32 × hidden_dim: b1]
    [f32 × hidden_dim: bn_gamma]
    [f32 × hidden_dim: bn_beta]
    [f32 × hidden_dim: bn_running_mean]
    [f32 × hidden_dim: bn_running_var]
    [f32 × n_bits × hidden_dim: W2]
    [f32 × n_bits: b2]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = 0xCAB1_DE3D
VERSION = 1


def parse_embedding_cache(path: Path) -> tuple[int, list[str], np.ndarray]:
    """解析 generate_embeddings.py 生成的二进制缓存。"""
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError("缓存文件过短")

    magic = struct.unpack(">I", data[0:4])[0]
    version = struct.unpack(">I", data[4:8])[0]
    dim = struct.unpack(">I", data[8:12])[0]
    _count = struct.unpack(">I", data[12:16])[0]

    if magic != 0xCAB1_EBED:
        raise ValueError(f"magic 不匹配: 0x{magic:08X}")
    if version != 1:
        raise ValueError(f"version 不支持: {version}")

    offset = 16
    words: list[str] = []
    vectors: list[np.ndarray] = []

    while offset < len(data):
        word_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        word = data[offset : offset + word_len].decode("utf-8")
        offset += word_len

        raw = np.frombuffer(data[offset : offset + dim * 4], dtype=np.uint8)
        vec = np.frombuffer(raw.tobytes(), dtype=">f4").copy().astype(np.float32)
        vectors.append(vec)
        words.append(word)
        offset += dim * 4

    return dim, words, np.vstack(vectors)


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for sign function.

    前向传播返回 sign(x)（+1 / -1），反向传播直接回传梯度。
    """
    return x + (torch.sign(x) - x).detach()


class DeepHashMLP(nn.Module):
    """深度语义哈希 MLP：dim -> hidden -> n_bits。"""

    def __init__(self, dim: int, hidden_dim: int, n_bits: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_bits, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        u = self.fc2(h)
        return u


def build_neighborhood_masks(
    X: np.ndarray,
    pos_k: int = 10,
    neg_k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """基于余弦相似度构建正负样本 mask。"""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = X_norm @ X_norm.T
    np.fill_diagonal(sim, -2.0)

    N = X.shape[0]
    pos_mask = np.zeros((N, N), dtype=bool)
    neg_mask = np.zeros((N, N), dtype=bool)

    pos_indices = np.argsort(-sim, axis=1)[:, :pos_k]
    neg_indices = np.argsort(sim, axis=1)[:, :neg_k]

    for i in range(N):
        pos_mask[i, pos_indices[i]] = True
        neg_mask[i, neg_indices[i]] = True

    return pos_mask, neg_mask


def build_hard_negatives(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_k: int,
) -> torch.Tensor:
    """根据当前连续投影动态挖掘 hardest negatives。"""
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.T  # (N, N)
    N = sim.shape[0]
    sim = sim.masked_fill(torch.eye(N, device=sim.device, dtype=torch.bool), -1e9)

    # 对每个 anchor，取最相似的非正样本作为负样本
    neg_sim = sim.masked_fill(pos_mask, -1e9)
    _, neg_indices = torch.topk(neg_sim, k=min(neg_k, N - 1), dim=1)
    neg_mask = torch.zeros_like(pos_mask)
    neg_mask.scatter_(1, neg_indices, True)
    return neg_mask


def info_nce_loss(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """标准 InfoNCE loss（在二进制/连续空间保持邻域结构）。"""
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.T / temperature

    N = sim.shape[0]
    self_mask = torch.eye(N, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)

    pos_sim = sim.masked_fill(~pos_mask, -1e9)
    pos_logsum = torch.logsumexp(pos_sim, dim=1)

    neg_sim = sim.masked_fill(~neg_mask, -1e9)
    neg_logsum = torch.logsumexp(neg_sim, dim=1)

    denom = torch.logsumexp(torch.stack([pos_logsum, neg_logsum], dim=1), dim=1)
    loss = -pos_logsum + denom

    valid = pos_mask.sum(dim=1) > 0
    return loss[valid].mean()


def pairwise_mse_loss(
    b: torch.Tensor,
    target_sim: torch.Tensor,
) -> torch.Tensor:
    """让二进制码的内积逼近真实 embedding 余弦相似度。

    b: (N, n_bits) in {-1, +1}
    target_sim: (N, N) cosine similarity in [-1, 1]
    """
    pred_sim = b @ b.T / b.shape[1]  # 归一化到 [-1, 1]
    return F.mse_loss(pred_sim, target_sim)


def train_deep_hash_v3(
    X: np.ndarray,
    n_bits: int = 52,
    hidden_dim: int = 512,
    epochs: int = 500,
    lr: float = 1e-3,
    pos_k: int = 10,
    neg_k: int = 50,
    temperature: float = 0.07,
    lambda_quant: float = 1.0,
    lambda_balance: float = 0.5,
    lambda_pair: float = 0.0,
    hard_neg_every: int = 10,
    seed: int = 42,
    teacher_X: np.ndarray | None = None,
) -> tuple[np.ndarray, nn.Module, list[float]]:
    """训练 DeepHash v3 MLP。

    若提供 teacher_X，则目标余弦相似度矩阵由教师向量计算，实现蒸馏。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, dim = X.shape
    device = torch.device("cpu")

    mean = X.mean(axis=0)
    Xc = X - mean
    X_norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)

    model = DeepHashMLP(dim, hidden_dim, n_bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(Xc.astype(np.float32)).to(device)

    # 教师向量用于计算目标相似度
    sim_source = teacher_X if teacher_X is not None else X
    sim_mean = sim_source.mean(axis=0)
    sim_source_c = sim_source - sim_mean
    sim_norm = sim_source_c / (np.linalg.norm(sim_source_c, axis=1, keepdims=True) + 1e-8)
    target_sim = torch.from_numpy((sim_norm @ sim_norm.T).astype(np.float32)).to(device)

    pos_mask_np, _ = build_neighborhood_masks(sim_source, pos_k, neg_k)
    pos_mask = torch.from_numpy(pos_mask_np).to(device)

    losses: list[float] = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        u = model(X_tensor)  # (N, n_bits) 连续投影
        b = ste_sign(u)  # (N, n_bits) 二进制 {-1, +1}

        # 动态 harder negatives
        if epoch % hard_neg_every == 0:
            neg_mask = build_hard_negatives(u.detach(), pos_mask, neg_k)
        else:
            neg_mask = build_hard_negatives(u.detach(), pos_mask, neg_k)

        loss_info = info_nce_loss(u, pos_mask, neg_mask, temperature)
        loss_quant = (u - b).abs().mean()
        bit_mean = (b + 1.0).mean(dim=0) / 2.0  # [0, 1]
        loss_balance = ((bit_mean - 0.5) ** 2).mean()
        loss_pair = pairwise_mse_loss(b, target_sim)

        # lambda_quant 退火：早期注重邻域结构，后期加强量化约束
        current_lambda_quant = 1.0 + (lambda_quant - 1.0) * min(epoch / (epochs * 0.5), 1.0)

        loss = (
            loss_info
            + current_lambda_quant * loss_quant
            + lambda_balance * loss_balance
            + lambda_pair * loss_pair
        )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"[epoch {epoch + 1:04d}] total={loss.item():.4f} "
                f"info={loss_info.item():.4f} quant={loss_quant.item():.4f} "
                f"balance={loss_balance.item():.4f} pair={loss_pair.item():.4f} "
                f"lambda_quant={current_lambda_quant:.2f}",
                file=sys.stderr,
            )

    return mean, model, losses


def load_deep_hash_model(path: Path) -> tuple[np.ndarray, nn.Module]:
    """从二进制文件加载 DeepHash v3 模型，返回 (mean, model)。"""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("DeepHash 缓存过短")
    magic = struct.unpack(">I", data[0:4])[0]
    version = struct.unpack(">I", data[4:8])[0]
    if magic != MAGIC:
        raise ValueError(f"magic 不匹配: 0x{magic:08X}")
    if version != VERSION:
        raise ValueError(f"version 不支持: {version}")
    dim = struct.unpack(">I", data[8:12])[0]
    hidden_dim = struct.unpack(">I", data[12:16])[0]
    n_bits = struct.unpack(">I", data[16:20])[0]

    offset = 20
    def read_f32s(n: int) -> np.ndarray:
        nonlocal offset
        arr = np.frombuffer(data[offset:offset + n * 4], dtype=">f4").copy().astype(np.float32)
        offset += n * 4
        return arr

    mean = read_f32s(dim)
    W1 = read_f32s(hidden_dim * dim).reshape(hidden_dim, dim)
    b1 = read_f32s(hidden_dim)
    gamma = read_f32s(hidden_dim)
    beta = read_f32s(hidden_dim)
    running_mean = read_f32s(hidden_dim)
    running_var = read_f32s(hidden_dim)
    W2 = read_f32s(n_bits * hidden_dim).reshape(n_bits, hidden_dim)
    b2 = read_f32s(n_bits)

    model = DeepHashMLP(dim, hidden_dim, n_bits)
    state = {
        "fc1.weight": torch.from_numpy(W1),
        "fc1.bias": torch.from_numpy(b1),
        "bn1.weight": torch.from_numpy(gamma),
        "bn1.bias": torch.from_numpy(beta),
        "bn1.running_mean": torch.from_numpy(running_mean),
        "bn1.running_var": torch.from_numpy(running_var),
        "fc2.weight": torch.from_numpy(W2),
        "fc2.bias": torch.from_numpy(b2),
    }
    model.load_state_dict(state)
    model.eval()
    return mean, model


def export_deep_hash_v3(
    output: Path,
    mean: np.ndarray,
    model: nn.Module,
) -> None:
    """导出 DeepHash v3 MLP 权重到二进制文件。"""
    model.eval()
    with torch.no_grad():
        state = model.state_dict()
        dim = state["fc1.weight"].shape[1]
        hidden_dim = state["fc1.weight"].shape[0]
        n_bits = state["fc2.weight"].shape[0]

        W1 = state["fc1.weight"].cpu().numpy().astype(np.float32)  # (hidden, dim)
        b1 = state["fc1.bias"].cpu().numpy().astype(np.float32)  # (hidden,)
        gamma = state["bn1.weight"].cpu().numpy().astype(np.float32)  # (hidden,)
        beta = state["bn1.bias"].cpu().numpy().astype(np.float32)  # (hidden,)
        running_mean = state["bn1.running_mean"].cpu().numpy().astype(np.float32)
        running_var = state["bn1.running_var"].cpu().numpy().astype(np.float32)
        W2 = state["fc2.weight"].cpu().numpy().astype(np.float32)  # (n_bits, hidden)
        b2 = state["fc2.bias"].cpu().numpy().astype(np.float32)  # (n_bits,)

    buf = bytearray()
    buf.extend(struct.pack(">I", MAGIC))
    buf.extend(struct.pack(">I", VERSION))
    buf.extend(struct.pack(">I", dim))
    buf.extend(struct.pack(">I", hidden_dim))
    buf.extend(struct.pack(">I", n_bits))

    for v in mean.astype(np.float32):
        buf.extend(struct.pack(">f", float(v)))
    for row in W1:
        for v in row:
            buf.extend(struct.pack(">f", float(v)))
    for v in b1:
        buf.extend(struct.pack(">f", float(v)))
    for v in gamma:
        buf.extend(struct.pack(">f", float(v)))
    for v in beta:
        buf.extend(struct.pack(">f", float(v)))
    for v in running_mean:
        buf.extend(struct.pack(">f", float(v)))
    for v in running_var:
        buf.extend(struct.pack(">f", float(v)))
    for row in W2:
        for v in row:
            buf.extend(struct.pack(">f", float(v)))
    for v in b2:
        buf.extend(struct.pack(">f", float(v)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buf)
    print(f"[write] {output} ({len(buf)} bytes)", file=sys.stderr)


def evaluate(
    X: np.ndarray,
    mean: np.ndarray,
    model: nn.Module,
    n_bits: int,
) -> None:
    """简单评估唯一码数量。"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy((X - mean).astype(np.float32))
        u = model(X_tensor)
        codes = (u >= 0).cpu().numpy().astype(np.uint8)
        # 52 bit 打包到 7 字节（取前 52 位）
        packed = np.packbits(codes, axis=1, bitorder="big")
        sims = np.fromiter(
            (int.from_bytes(row.tobytes()[:7], "big") for row in packed),
            dtype=np.uint64,
            count=packed.shape[0],
        )
        unique = len(np.unique(sims))
        print(f"[eval] 量化后唯一码数量: {unique} / {X.shape[0]}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端语义哈希训练 v3（HSH-64，MLP + STE）")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        required=True,
        help="embedding 缓存路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("deep_hash_v3_64.bin"),
        help="输出二进制路径",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=52,
        help="哈希码位数（默认 52）",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="MLP 隐藏层维度（默认 512）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--pos-k",
        type=int,
        default=10,
        help="每个 anchor 的正样本数",
    )
    parser.add_argument(
        "--neg-k",
        type=int,
        default=50,
        help="每个 anchor 的负样本数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="InfoNCE 温度系数",
    )
    parser.add_argument(
        "--lambda-quant",
        type=float,
        default=1.0,
        help="量化损失权重",
    )
    parser.add_argument(
        "--lambda-balance",
        type=float,
        default=0.5,
        help="比特平衡损失权重",
    )
    parser.add_argument(
        "--lambda-pair",
        type=float,
        default=0.0,
        help="成对内积损失权重（默认关闭，避免小样本过拟合）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=None,
        help="教师向量缓存（用于蒸馏，默认与 embedding-cache 相同）",
    )
    args = parser.parse_args()

    dim, words, X = parse_embedding_cache(args.embedding_cache)
    print(
        f"[train] 加载学生 {X.shape[0]} 个 {dim}-dim 向量",
        file=sys.stderr,
    )

    teacher_X = None
    if args.teacher_cache is not None:
        _, _, teacher_X = parse_embedding_cache(args.teacher_cache)
        print(
            f"[train] 加载教师 {teacher_X.shape[0]} 个 {teacher_X.shape[1]}-dim 向量",
            file=sys.stderr,
        )
        if teacher_X.shape[0] != X.shape[0]:
            raise ValueError("教师向量与学生向量数量不一致")

    mean, model, losses = train_deep_hash_v3(
        X,
        n_bits=args.n_bits,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        pos_k=args.pos_k,
        neg_k=args.neg_k,
        temperature=args.temperature,
        lambda_quant=args.lambda_quant,
        lambda_balance=args.lambda_balance,
        lambda_pair=args.lambda_pair,
        seed=args.seed,
        teacher_X=teacher_X,
    )

    export_deep_hash_v3(args.output, mean, model)
    evaluate(X, mean, model, args.n_bits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
