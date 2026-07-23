#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSH-64 端到端测试：生成 mock embedding → 训练 PCA → 调用 Rust 测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_embeddings import write_cache
from train_pca_64 import train_pca, write_pca_cache


def generate_mock_vocab(path, n=50):
    """生成测试词表。"""
    words = [f"词{i:03d}" for i in range(n)]
    words += ["苹果", "香蕉", "橙子", "水果", "汽车", "火车", "飞机"]
    path.write_text("\n".join(words), encoding="utf-8")
    return words


def generate_mock_embeddings(words, dim=64, output=None):
    """生成随机但确定性的 mock embedding。"""
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(len(words), dim)).astype(np.float32)
    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    items = [(w, v) for w, v in zip(words, vectors)]
    if output:
        write_cache(output, dim, items)
    return np.array(vectors)


def main():
    root = Path(__file__).parent.parent
    data_dir = root / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = data_dir / "vocab.txt"
    emb_path = data_dir / "embedding.cache"
    pca_path = data_dir / "pca_52.bin"

    print("[1/4] 生成测试词表...", file=sys.stderr)
    words = generate_mock_vocab(vocab_path)

    print("[2/4] 生成 mock embedding...", file=sys.stderr)
    generate_mock_embeddings(words, dim=64, output=emb_path)

    print("[3/4] 训练 52 维 PCA...", file=sys.stderr)
    X = np.loadtxt(str(emb_path) + ".npy") if False else None
    # 重新读取缓存训练 PCA
    from train_pca_64 import parse_embedding_cache
    dim, _, X = parse_embedding_cache(emb_path)
    mean, components, evr = train_pca(X)
    write_pca_cache(pca_path, dim, mean, components, evr)

    print("[4/4] 运行 Rust 端到端测试...", file=sys.stderr)
    result = subprocess.run(
        ["cargo", "test", "--", "--nocapture"],
        cwd=root,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
