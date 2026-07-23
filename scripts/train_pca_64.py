#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 HSH-64 专用 PCA，输出 52 维投影矩阵。

二进制格式（与 Rust PcaProjection::from_bytes 兼容）：
    [u32: magic = 0xCAB1_3CA2]
    [u32: version = 1]
    [u32: dim]
    [u32: n_components = 52]
    [f32 × dim: mean]
    [f32 × 52 × dim: components]
    [f32 × 52: explained_variance_ratio]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

PCA_MAGIC = 0xCAB1_3CA2
PCA_VERSION = 1
N_COMPONENTS = 52


def parse_embedding_cache(path):
    """解析 embedding 缓存，返回 (dim, words, X)。"""
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError("缓存文件过短")

    magic = struct.unpack(">I", data[0:4])[0]
    if magic != 0xCAB1_EBED:
        raise ValueError(f"magic 不匹配: 0x{magic:08X}")
    version = struct.unpack(">I", data[4:8])[0]
    if version != 1:
        raise ValueError(f"version 不支持: {version}")
    dim = struct.unpack(">I", data[8:12])[0]
    vocab_size = struct.unpack(">I", data[12:16])[0]

    words = []
    vectors = np.zeros((vocab_size, dim), dtype=np.float32)
    offset = 16
    for i in range(vocab_size):
        wlen = struct.unpack(">H", data[offset:offset + 2])[0]
        offset += 2
        words.append(data[offset:offset + wlen].decode("utf-8"))
        offset += wlen
        raw = np.frombuffer(data[offset:offset + dim * 4], dtype=np.uint8)
        vectors[i] = np.frombuffer(raw.tobytes(), dtype=">f4").copy().astype(np.float32)
        offset += dim * 4

    return dim, words, vectors


def train_pca(X, n_components=N_COMPONENTS):
    """训练 PCA 并返回 (mean, components, evr)。"""
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise RuntimeError("需要安装 scikit-learn") from e

    mean = X.mean(axis=0)
    Xc = X - mean
    actual = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=actual, random_state=42)
    pca.fit(Xc)

    components = np.zeros((n_components, X.shape[1]), dtype=np.float32)
    components[:actual] = pca.components_.astype(np.float32)
    evr = np.zeros(n_components, dtype=np.float32)
    evr[:actual] = pca.explained_variance_ratio_.astype(np.float32)

    return mean.astype(np.float32), components, evr


def write_pca_cache(path, dim, mean, components, evr):
    """写入 PCA 二进制缓存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_components = components.shape[0]
    with path.open("wb") as f:
        f.write(struct.pack(">I", PCA_MAGIC))
        f.write(struct.pack(">I", PCA_VERSION))
        f.write(struct.pack(">I", dim))
        f.write(struct.pack(">I", n_components))
        for v in mean:
            f.write(struct.pack(">f", float(v)))
        for row in components:
            for v in row:
                f.write(struct.pack(">f", float(v)))
        for v in evr:
            f.write(struct.pack(">f", float(v)))


def load_pca_cache(path):
    """读取 PCA 二进制缓存，返回 (mean, components, evr)。"""
    data = Path(path).read_bytes()
    if len(data) < 16:
        raise ValueError("PCA 缓存过短")
    magic = struct.unpack(">I", data[0:4])[0]
    version = struct.unpack(">I", data[4:8])[0]
    dim = struct.unpack(">I", data[8:12])[0]
    n_components = struct.unpack(">I", data[12:16])[0]
    if magic != PCA_MAGIC:
        raise ValueError(f"PCA magic 不匹配: 0x{magic:08X}")
    if version != PCA_VERSION:
        raise ValueError(f"PCA version 不支持: {version}")
    if n_components != N_COMPONENTS:
        raise ValueError(f"PCA 组件数错误: {n_components}")

    offset = 16
    mean = np.frombuffer(data[offset:offset + dim * 4], dtype=">f4").copy().astype(np.float32)
    offset += dim * 4
    components = np.frombuffer(
        data[offset:offset + n_components * dim * 4], dtype=">f4"
    ).copy().astype(np.float32).reshape(n_components, dim)
    offset += n_components * dim * 4
    evr = np.frombuffer(data[offset:offset + n_components * 4], dtype=">f4").copy().astype(np.float32)
    return mean, components, evr


def main():
    parser = argparse.ArgumentParser(description="训练 HSH-64 PCA（52 维）")
    parser.add_argument("--embedding-cache", required=True, type=Path, help="embedding 缓存路径")
    parser.add_argument("-o", "--output", required=True, type=Path, help="输出 PCA 缓存路径")
    args = parser.parse_args()

    dim, words, X = parse_embedding_cache(args.embedding_cache)
    print(f"加载 embedding: {len(words)} 个 {dim} 维向量", file=sys.stderr)

    mean, components, evr = train_pca(X)
    write_pca_cache(args.output, dim, mean, components, evr)
    print(f"完成：PCA 已写入 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
