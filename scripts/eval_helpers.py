# -*- coding: utf-8 -*-
"""
评估脚本的共享辅助函数。

为什么单独抽出来：
    52_eval_pipeline.py 与 55_distance_stats.py 都要
    「读导出模型」「把教师嵌入 PCA 到学生空间」这两件事。
    复制两份会导致两处行为漂移（本项目已经吃过「两份映射表不一致」的亏）。

注意：DeepHash 二进制格式与 Rust 侧 src/deep_hash.rs 完全一致，
     这里的读取实现必须与那边同步 —— 改动一处要改另一处。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))


def load_exported(path: Path):
    """读 export_deep_hash 写出的二进制，返回 (mean, model)。

    格式（big-endian）：
        magic(u32) version(u32) dim(u32) hidden(u32) n_bits(u32)
        mean[dim]
        W1[hidden*dim] b1[hidden] gamma[hidden] beta[hidden]
        running_mean[hidden] running_var[hidden]
        W2[n_bits*hidden] b2[n_bits]
    """
    import struct

    import torch

    from cabinet_bnn.hsh.deep_hash import DeepHashMLP

    data = Path(path).read_bytes()
    magic, ver, dim, hidden, n_bits = struct.unpack(">5I", data[:20])
    off = 20

    def f32(cnt: int) -> np.ndarray:
        nonlocal off
        a = np.frombuffer(data[off:off + cnt * 4], dtype=">f4").copy().astype(np.float32)
        off += cnt * 4
        return a

    mean = f32(dim)
    W1 = f32(hidden * dim).reshape(hidden, dim)
    b1 = f32(hidden)
    gamma = f32(hidden)
    beta = f32(hidden)
    rm = f32(hidden)
    rv = f32(hidden)
    W2 = f32(n_bits * hidden).reshape(n_bits, hidden)
    b2 = f32(n_bits)

    model = DeepHashMLP(dim, hidden, n_bits)
    model.load_state_dict({
        "fc1.weight": torch.from_numpy(W1), "fc1.bias": torch.from_numpy(b1),
        "bn1.weight": torch.from_numpy(gamma), "bn1.bias": torch.from_numpy(beta),
        "bn1.running_mean": torch.from_numpy(rm),
        "bn1.running_var": torch.from_numpy(rv),
        "fc2.weight": torch.from_numpy(W2), "fc2.bias": torch.from_numpy(b2),
    })
    model.eval()
    return mean, model


def load_pca_student(teacher: np.ndarray, dim: int,
                     basis_path: Path | None = None) -> np.ndarray:
    """把教师嵌入 PCA 降到 dim 维并 L2 归一化。

    若给了 basis_path 且文件存在，复用已有的 mean/components ——
    这样不同词表映射到【同一个】子空间，比较才成立。
    """
    from cabinet_bnn.data.embedding_cache import normalize_rows

    if dim <= 0 or dim >= teacher.shape[1]:
        return teacher

    if basis_path and Path(basis_path).exists():
        d = np.load(basis_path)
        mean, comps = d["mean"], d["components"]
        if comps.shape[0] != dim:
            raise ValueError(f"PCA 基维数 {comps.shape[0]} != 请求 {dim}")
        return normalize_rows(((teacher - mean) @ comps.T).astype(np.float32))

    mean = teacher.mean(axis=0, keepdims=True).astype(np.float32)
    xc = teacher - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:dim].astype(np.float32)
    return normalize_rows((xc @ comps.T).astype(np.float32))
