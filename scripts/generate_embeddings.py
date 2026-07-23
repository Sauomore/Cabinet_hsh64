#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 sentence-transformers embedding 缓存（HSH-64 格式）。

二进制格式：
    [u32: magic = 0xCAB1_EBED]
    [u32: version = 1]
    [u32: dim]
    [u32: vocab_size]
    for each word:
      [u16: len(word_bytes)]
      [bytes: UTF-8 word]
      [f32 × dim: vector]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

EMBEDDING_MAGIC = 0xCAB1_EBED
EMBEDDING_VERSION = 1
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def encode_words(words, model_name, local_model=None, batch_size=128):
    """用词表生成 embedding 向量。"""
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
    return dim, [(w, v.astype(np.float32)) for w, v in zip(words, vectors)]


def write_cache(path, dim, items):
    """写入二进制缓存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack(">I", EMBEDDING_MAGIC))
        f.write(struct.pack(">I", EMBEDDING_VERSION))
        f.write(struct.pack(">I", dim))
        f.write(struct.pack(">I", len(items)))
        for word, vec in items:
            wb = word.encode("utf-8")
            f.write(struct.pack(">H", len(wb)))
            f.write(wb)
            for v in vec:
                f.write(struct.pack(">f", float(v)))


def main():
    parser = argparse.ArgumentParser(description="生成 embedding 缓存")
    parser.add_argument("--vocab", required=True, type=Path, help="词表文件路径")
    parser.add_argument("-o", "--output", required=True, type=Path, help="输出缓存路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名称（默认：{DEFAULT_MODEL}）")
    parser.add_argument("--local-model", type=Path, help="本地模型路径")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    words = [line.strip() for line in args.vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"待编码词数: {len(words)}", file=sys.stderr)

    dim, items = encode_words(words, args.model, args.local_model, args.batch_size)
    write_cache(args.output, dim, items)
    print(f"完成：已写入 {len(items)} 个 {dim}-dim 向量到 {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
