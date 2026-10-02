# -*- coding: utf-8 -*-
"""
把 pytorch_model.bin 转成 model.safetensors。

为什么需要：
    本机 torch 2.5.1，而 transformers 5.13 因 CVE-2025-32434
    拒绝用 torch.load 加载 .bin（要求 torch >= 2.6）。
    safetensors 不受此限制，而且加载更快、可分片映射。

    这一步只做格式转换，不改任何权重值。

用法：
    python scripts/44_bin_to_safetensors.py --dir F:\\bge-m3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="bin -> safetensors")
    ap.add_argument("--dir", type=Path, required=True, help="模型目录")
    ap.add_argument("--keep-bin", action="store_true", help="转换后保留原 .bin")
    args = ap.parse_args()

    src = args.dir / "pytorch_model.bin"
    dst = args.dir / "model.safetensors"
    if not src.exists():
        print(f"找不到 {src}")
        return 1

    print("=" * 68)
    print("转换为 safetensors")
    print("=" * 68)
    print(f"  源   {src}  ({human(src.stat().st_size)})")
    print(f"  目标 {dst}")

    import torch
    from safetensors.torch import save_file

    t0 = time.time()
    print("\n  读取 .bin …")
    sd = torch.load(str(src), map_location="cpu", weights_only=True, mmap=True)
    print(f"    {len(sd)} 个张量，用时 {time.time()-t0:.1f}s")

    # safetensors 要求张量连续且无共享存储
    print("  规整张量（contiguous）…")
    clean = {}
    for k, v in sd.items():
        clean[k] = v.contiguous().clone() if not v.is_contiguous() else v
    del sd

    print("  写入 safetensors …")
    t1 = time.time()
    save_file(clean, str(dst), metadata={"format": "pt",
                                         "converted_from": "pytorch_model.bin"})
    del clean
    print(f"    完成，用时 {time.time()-t1:.1f}s")
    print(f"    大小 {human(dst.stat().st_size)}")

    if not args.keep_bin:
        freed = src.stat().st_size
        src.unlink()
        print(f"\n  已删除原 .bin，释放 {human(freed)}")
    else:
        print("\n  --keep-bin：保留原 .bin")

    print(f"\n  总用时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
