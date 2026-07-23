#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成独立的 reranker embedding 缓存（用于 HSH-64 双 embedding 精排）。

双 embedding 设计：
- 编码 embedding（encoder embedding）：轻量模型如 bge-small，用于生成 HSH-64 码和粗排
- 精排 embedding（reranker embedding）：更强的模型如 bge-large，用于最终候选重排

用法：
    F:\\python311\\python.exe scripts/generate_reranker_embeddings_64.py \\
        --vocab tests/data/bench_vocab.txt \\
        -o tests/data/reranker_embedding.cache \\
        --local-model i:\\path\\to\\bge-large-zh-v1.5

二进制格式与 generate_embeddings.py 一致，Rust 端可直接用 FileCachedEmbedding 加载。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import generate_embeddings

DEFAULT_RERANKER_MODEL = "BAAI/bge-large-zh-v1.5"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 HSH-64 独立 reranker embedding 缓存")
    parser.add_argument("--vocab", required=True, type=Path, help="词表文件路径")
    parser.add_argument("-o", "--output", required=True, type=Path, help="输出缓存路径")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL, help=f"模型名称（默认：{DEFAULT_RERANKER_MODEL}）")
    parser.add_argument("--local-model", type=Path, help="本地模型路径（推荐，避免联网下载）")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    words = [line.strip() for line in args.vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[reranker] 待编码词数: {len(words)}", file=sys.stderr)

    dim, items = generate_embeddings.encode_words(
        words,
        args.model,
        args.local_model,
        args.batch_size,
    )
    generate_embeddings.write_cache(args.output, dim, items)
    print(f"[reranker] 完成：已写入 {len(items)} 个 {dim}-dim 向量到 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
