#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较两个缓存文件的词集合差异。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_words(p: Path):
    with p.open("rb") as f:
        magic, ver, dim, n = struct.unpack(">IIII", f.read(16))
        words = []
        for _ in range(n):
            l = struct.unpack(">H", f.read(2))[0]
            words.append(f.read(l).decode("utf-8"))
            f.read(4 * dim)
        return words


if __name__ == "__main__":
    w1 = set(read_words(Path(sys.argv[1])))
    w2 = set(read_words(Path(sys.argv[2])))
    print("emb only:", len(w1 - w2), list(w1 - w2)[:10])
    print("rer only:", len(w2 - w1), list(w2 - w1)[:10])
