#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 bge-small 输入蒸馏已训练好的 HSH 码（来自 bge-large 模型）。

目标：让轻量 MLP 从 bge-small 直接预测强模型生成的 52-bit 码，
部署时只需 bge-small + 小 MLP，无需 bge-large。
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

sys.path.insert(0, str(Path(__file__).parent))
import train_deep_hash_v3_64
from train_pca_64 import parse_embedding_cache

MAGIC = 0xCAB1_DE3D
VERSION = 1


class DeepHashMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_bits: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_bits, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(h)


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.sign(x) - x).detach()


def train_code_distillation(
    X: np.ndarray,
    target_codes: np.ndarray,
    hidden_dim: int,
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, nn.Module]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, dim = X.shape
    n_bits = target_codes.shape[1]
    device = torch.device("cpu")

    mean = X.mean(axis=0)
    Xc = X - mean

    model = DeepHashMLP(dim, hidden_dim, n_bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(Xc.astype(np.float32)).to(device)
    # target: {0,1} -> {-1,+1}
    target_b = torch.from_numpy((target_codes.astype(np.float32) * 2.0 - 1.0)).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        u = model(X_tensor)
        b = ste_sign(u)

        # BCE 风格：让 u 接近目标 sign
        loss_match = F.mse_loss(u, target_b)
        loss_quant = (u - b).abs().mean()
        bit_mean = (b + 1.0).mean(dim=0) / 2.0
        loss_balance = ((bit_mean - 0.5) ** 2).mean()

        loss = loss_match + loss_quant + 0.5 * loss_balance
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"[epoch {epoch + 1:04d}] total={loss.item():.4f} "
                f"match={loss_match.item():.4f} quant={loss_quant.item():.4f} "
                f"balance={loss_balance.item():.4f}",
                file=sys.stderr,
            )

    return mean, model


def export_model(output: Path, mean: np.ndarray, model: nn.Module) -> None:
    model.eval()
    with torch.no_grad():
        state = model.state_dict()
        dim = state["fc1.weight"].shape[1]
        hidden_dim = state["fc1.weight"].shape[0]
        n_bits = state["fc2.weight"].shape[0]

        W1 = state["fc1.weight"].cpu().numpy().astype(np.float32)
        b1 = state["fc1.bias"].cpu().numpy().astype(np.float32)
        gamma = state["bn1.weight"].cpu().numpy().astype(np.float32)
        beta = state["bn1.bias"].cpu().numpy().astype(np.float32)
        running_mean = state["bn1.running_mean"].cpu().numpy().astype(np.float32)
        running_var = state["bn1.running_var"].cpu().numpy().astype(np.float32)
        W2 = state["fc2.weight"].cpu().numpy().astype(np.float32)
        b2 = state["fc2.bias"].cpu().numpy().astype(np.float32)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", type=Path, required=True, help="学生输入 (bge-small)")
    parser.add_argument("--teacher-model", type=Path, required=True, help="教师 Deep Hash 模型 (bge-large 输入)")
    parser.add_argument("--teacher-cache", type=Path, required=True, help="教师模型输入缓存 (bge-large)")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载学生输入
    _, words, X = parse_embedding_cache(args.input_cache)
    # 加载教师输入并生成目标码
    _, t_words, t_X = parse_embedding_cache(args.teacher_cache)
    if words != t_words:
        w2i = {w: i for i, w in enumerate(t_words)}
        order = [w2i[w] for w in words]
        t_X = t_X[order]

    mean_t, teacher_model = train_deep_hash_v3_64.load_deep_hash_model(args.teacher_model)
    teacher_model.eval()
    with torch.no_grad():
        u_t = teacher_model(torch.from_numpy((t_X - mean_t).astype(np.float32)))
        target_codes = (u_t >= 0).cpu().numpy().astype(np.uint8)

    print(f"[distill] 目标码唯一数: {len(np.unique(target_codes @ (1 << np.arange(52))))}", file=sys.stderr)

    mean, model = train_code_distillation(X, target_codes, args.hidden_dim, args.epochs, seed=args.seed)
    export_model(args.output, mean, model)


if __name__ == "__main__":
    main()
