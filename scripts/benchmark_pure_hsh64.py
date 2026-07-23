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
import generate_embeddings
from train_pca_64 import parse_embedding_cache, train_pca, write_pca_cache
import train_deep_hash_v3_64

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
    parser.add_argument("--reranker-model", default="BAAI/bge-large-zh-v1.5", help="reranker 模型名称")
    parser.add_argument("--local-reranker-model", type=Path, help="本地 reranker 模型路径")
    parser.add_argument("--reranker-cache", type=Path, default=Path("tests/data/reranker_embedding.cache"), help="reranker embedding 缓存路径")
    parser.add_argument("--skip-reranker", action="store_true", help="不生成/使用 reranker embedding")
    parser.add_argument("--segment-counts", default="1,2,4,13", help="MIH 段数，逗号分隔")
    parser.add_argument("--radii", default="0,2,4,6,8", help="Hamming radius，逗号分隔")
    parser.add_argument("--dim", type=int, default=384, help="mock embedding 维度")
    parser.add_argument("--asymmetric", action="store_true", help="使用非对称距离粗排")
    parser.add_argument("--deep-hash", action="store_true", help="使用 Deep Hash v3 替代 PCA")
    parser.add_argument("--deep-hash-path", type=Path, default=Path("tests/data/deep_hash_v3_64.bin"), help="Deep Hash 模型路径")
    parser.add_argument("--skip-deep-hash-train", action="store_true", help="跳过 Deep Hash 训练（复用已有模型）")
    parser.add_argument("--deep-hash-epochs", type=int, default=500, help="Deep Hash 训练轮数")
    parser.add_argument("--deep-hash-hidden-dim", type=int, default=512, help="Deep Hash 隐藏层维度")
    parser.add_argument("--teacher-cache", type=Path, help="教师向量缓存（Deep Hash 蒸馏）")
    parser.add_argument("--sim-override", type=Path, help="后处理优化的 sim 码覆盖缓存路径")
    args = parser.parse_args()

    ensure_vocab(args.vocab, args.vocab_size)
    words = [line.strip() for line in args.vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"词表大小: {len(words)}", file=sys.stderr)

    emb_path = args.output / "embedding.cache"
    pca_path = args.output / "pca_52.bin"
    deep_hash_path = args.deep_hash_path
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. 生成/复用 embedding
    if emb_path.exists():
        print(f"复用已有 embedding: {emb_path}", file=sys.stderr)
        dim, _, _ = parse_embedding_cache(emb_path)
    else:
        print("生成 embedding...", file=sys.stderr)
        if args.use_mock:
            vectors = generate_mock_embeddings(words, args.dim)
            generate_embeddings.write_cache(emb_path, args.dim, list(zip(words, vectors)))
            dim = args.dim
        else:
            dim, vectors = generate_embeddings_with_model(words, args.model, args.local_model)
            generate_embeddings.write_cache(emb_path, dim, list(zip(words, vectors)))
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

    # 2.5 训练/复用 Deep Hash v3
    if args.deep_hash:
        if args.skip_deep_hash_train and deep_hash_path.exists():
            print(f"复用已有 Deep Hash: {deep_hash_path}", file=sys.stderr)
        else:
            print("训练 Deep Hash v3 (52-bit)...", file=sys.stderr)
            _, _, X = parse_embedding_cache(emb_path)
            teacher_X = None
            if args.teacher_cache is not None:
                _, _, teacher_X = parse_embedding_cache(args.teacher_cache)
            mean, model, _ = train_deep_hash_v3_64.train_deep_hash_v3(
                X,
                n_bits=52,
                hidden_dim=args.deep_hash_hidden_dim,
                epochs=args.deep_hash_epochs,
                teacher_X=teacher_X,
            )
            train_deep_hash_v3_64.export_deep_hash_v3(deep_hash_path, mean, model)
            train_deep_hash_v3_64.evaluate(X, mean, model, 52)
            print(f"完成: {deep_hash_path}", file=sys.stderr)

    # 3. 生成/复用 reranker embedding（双 embedding 精排）
    if args.skip_reranker:
        print("跳过 reranker embedding", file=sys.stderr)
        reranker_path = None
    elif args.reranker_cache.exists():
        print(f"复用已有 reranker: {args.reranker_cache}", file=sys.stderr)
        reranker_path = args.reranker_cache
    else:
        print("生成 reranker embedding...", file=sys.stderr)
        if args.use_mock:
            # mock 场景下：复用编码 embedding 作为 reranker（代码路径正确，效果等于单 embedding）
            import shutil
            shutil.copy(emb_path, args.reranker_cache)
            _, rd, _ = parse_embedding_cache(args.reranker_cache)
            print(f"[mock] reranker 复用编码 embedding，dim={rd}", file=sys.stderr)
        else:
            _, ritems = generate_embeddings.encode_words(
                words,
                args.reranker_model,
                args.local_reranker_model,
                128,
            )
            generate_embeddings.write_cache(args.reranker_cache, rd := len(ritems[0][1]), ritems)
            print(f"reranker 完成: dim={rd}, {len(ritems)} 个向量 -> {args.reranker_cache}", file=sys.stderr)
        reranker_path = args.reranker_cache

    # 4. 运行 Rust benchmark
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
    if args.deep_hash:
        cmd.extend(["--deep-hash", str(deep_hash_path)])
    if args.sim_override is not None:
        cmd.extend(["--sim-override", str(args.sim_override)])
    if reranker_path:
        cmd.extend(["--reranker", str(reranker_path)])
    if args.asymmetric:
        cmd.append("--asymmetric")
    print(" ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
