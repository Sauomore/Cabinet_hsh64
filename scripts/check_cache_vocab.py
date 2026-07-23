#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查缓存词表顺序是否与文本词表一致。"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def read_words(path: Path, max_n: int = 0) -> list[str]:
    with path.open("rb") as f:
        magic, ver, dim, n = struct.unpack(">IIII", f.read(16))
        print(f"magic={hex(magic)}, ver={ver}, dim={dim}, n={n}", file=sys.stderr)
        words = []
        for i in range(n):
            l = struct.unpack(">H", f.read(2))[0]
            words.append(f.read(l).decode("utf-8"))
            f.read(4 * dim)
            if max_n and i + 1 >= max_n:
                break
        return words


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    words = read_words(cache, max_n=10)
    print("\n".join(words))
