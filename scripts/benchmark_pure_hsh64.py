#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 HSH-64 基准测试驱动脚本。

流程：
1. 生成/复用 embedding 缓存
2. 训练/复用 52 维 PCA 缓存
3. 调用 Rust benchmark example

用法：
    F:\python311\python.exe scripts/benchmark_pure_hsh64.py \
        --vocab tests/data/vocab.txt \
        --output tests/data \
        --top-k 10 \
        --queries 100 \
        --use-mock

或使用真实模型：
    F:\python311\python.exe scripts/benchmark_pure_hsh64.py \
        --vocab tests/data/vocab.txt \
        --output tests/data \
        --model BAAI/bge-small-zh-v1.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_embeddings import write_cache
from train_pca_64 import parse_embedding_cache, train_pca, write_pca_cache

DEFAULT_VOCAB = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓", "樱桃", "柠檬",
                 "汽车", "火车", "飞机", "轮船", "自行车", "摩托车",
                 "快乐", "悲伤", "愤怒", "恐惧", "惊讶", "兴奋",
                 "学校", "医院", "银行", "商店", "公园", "图书馆", "博物馆",
                 "北京", "上海", "广州", "深圳", "杭州", "成都", "西安",
                 "猫", "狗", "鸟", "鱼", "老虎", "狮子", "大象"]


def ensure_vocab(path: Path, size: int = 200):
    """确保词表文件存在。"""
    if path.exists():
        return

    words = []
    categories = [
        ("水果", ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓", "樱桃", "柠檬", "桃子", "梨", "芒果", "菠萝"]),
        ("交通工具", ["汽车", "火车", "飞机", "轮船", "自行车", "摩托车", "公交车", "地铁", "高铁", "出租车"]),
        ("情绪", ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "兴奋", "焦虑", "满足", "失望", "希望"]),
        ("地点", ["学校", "医院", "银行", "商店", "公园", "图书馆", "博物馆", "餐厅", "酒店", "机场"]),
        ("城市", ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京", "武汉", "重庆"]),
        ("动物", ["猫", "狗", "鸟", "鱼", "老虎", "狮子", "大象", "熊猫", "猴子", "兔子"]),
    ]

    for item in [item for _, items in categories for item in items]:
        words.append(item)
        if len(words) >= size:
            break

    # 如果还不够，用序号补充
    extra_idx = 0
    while len(words) < size:
        words.append(f"词{extra_idx:04d}")
        extra_idx += 1

    # 去重并保证数量
    words = list(dict.fromkeys(words))[:size]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(words), encoding="utf-8")


def generate_mock_embeddings(words, dim=384):
    """生成结构化 mock embedding：同类词共享中心， PCA 后语义邻居 Hamming 距离近。"""
    rng = np.random.default_rng(42)
    categories = ["水果", "交通工具", "情绪", "地点", "城市", "动物", "其他"]
    keywords = {
        "水果": ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "草莓", "樱桃", "柠檬", "桃子", "梨", "芒果", "菠萝"],
        "交通工具": ["汽车", "火车", "飞机", "轮船", "自行车", "摩托车", "公交车", "地铁", "高铁", "出租车"],
        "情绪": ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "兴奋", "焦虑", "满足", "失望", "希望"],
        "地点": ["学校", "医院", "银行", "商店", "公园", "图书馆", "博物馆", "餐厅", "酒店", "机场"],
        "城市": ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "南京", "武汉", "重庆"],
        "动物": ["猫", "狗", "鸟", "鱼", "老虎", "狮子", "大象", "熊猫", "猴子", "兔子"],
    }

    # 预生成每个类别的中心向量
    category_centers = {}
    for cat in categories:
        center = rng.normal(size=dim).astype(np.float32)
        center /= np.linalg.norm(center)
        category_centers[cat] = center

    def category_of(w):
        for cat, items in keywords.items():
            if any(item in w for item in items):
                return cat
        return "其他"

    vectors = np.zeros((len(words), dim), dtype=np.float32)
    for i, w in enumerate(words):
        cat = category_of(w)
        center = category_centers[cat]
        noise = rng.normal(scale=0.08, size=dim).astype(np.float32)
        v = center + noise
        v /= np.maximum(np.linalg.norm(v), 1e-8)
        vectors[i] = v

    return vectors


def generate_embeddings_with_model(words, model_name, local_model=None, batch_size=128):
    """用 sentence-transformers 生成真实 embedding。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("需要安装 sentence-transformers") from e

    if local_model:
        model = SentenceTransformer(str(local_model))
    else:
        model = SentenceTransformer(model_name)

    dim = model.get_sentence_embedding_dimension()
    vectors = model.encode(words, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return dim, vectors.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="HSH-64 纯索引基准测试")
    parser.add_argument("--vocab", type=Path, default=Path("tests/data/bench_vocab.txt"), help="词表文件路径")
    parser.add_argument("--output", type=Path, default=Path("tests/data"), help="输出目录")
    parser.add_argument("--vocab-size", type=int, default=2000, help="自动生成词表的大小")
    parser.add_argument("--top-k", type=int, default=10, help="Recall@K 的 K")
    parser.add_argument("--queries", type=int, default=100, help="查询词数量")
    parser.add_argument("--use-mock", action="store_true", help="使用 mock embedding 而非真实模型")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="真实模型名称")
    parser.add_argument("--local-model", type=Path, help="本地模型路径")
    parser.add_argument("--segment-counts", default="1,2,4,13", help="MIH 段数，逗号分隔")
    parser.add_argument("--radii", default="0,2,4,6,8", help="Hamming radius，逗号分隔")
    parser.add_argument("--dim", type=int, default=384, help="mock embedding 维度")
    args = parser.parse_args()

    ensure_vocab(args.vocab, args.vocab_size)
    words = [line.strip() for line in args.vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"词表大小: {len(words)}", file=sys.stderr)

    emb_path = args.output / "embedding.cache"
    pca_path = args.output / "pca_52.bin"
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. 生成/复用 embedding
    if emb_path.exists():
        print(f"复用已有 embedding: {emb_path}", file=sys.stderr)
        dim, _, _ = parse_embedding_cache(emb_path)
    else:
        print("生成 embedding...", file=sys.stderr)
        if args.use_mock:
            vectors = generate_mock_embeddings(words, args.dim)
            write_cache(emb_path, args.dim, list(zip(words, vectors)))
            dim = args.dim
        else:
            dim, vectors = generate_embeddings_with_model(words, args.model, args.local_model)
            write_cache(emb_path, dim, list(zip(words, vectors)))
        print(f"完成: {emb_path}", file=sys.stderr)

    # 2. 训练/复用 PCA
    if pca_path.exists():
        print(f"复用已有 PCA: {pca_path}", file=sys.stderr)
    else:
        print("训练 52 维 PCA...", file=sys.stderr)
        _, _, X = parse_embedding_cache(emb_path)
        mean, components, evr = train_pca(X)
        write_pca_cache(pca_path, dim, mean, components, evr)
        print(f"完成: {pca_path}", file=sys.stderr)

    # 3. 运行 Rust benchmark
    print("\n运行 Rust benchmark...", file=sys.stderr)
    cmd = [
        "cargo", "run", "--example", "benchmark_pure_hsh64", "--release", "--",
        "--embedding", str(emb_path),
        "--pca", str(pca_path),
        "--top-k", str(args.top_k),
        "--queries", str(args.queries),
        "--segment-counts", args.segment_counts,
        "--radii", args.radii,
    ]
    print(" ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
