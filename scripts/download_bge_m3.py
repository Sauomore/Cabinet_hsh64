# -*- coding: utf-8 -*-
"""
直接从 hf-mirror 下载 bge-m3 到本地目录。

为什么不用 sentence_transformers 自动下载：
    huggingface_hub 的重试/超时逻辑不透明，卡住时没有输出，
    无法判断是「慢」还是「死了」。这里手工下载，逐文件报进度与速度。

用法：
    python download_bge_m3.py --out F:\\bge-m3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
REPO = "BAAI/bge-m3"

#: 只需要这些文件（跳过 onnx/，那是给 onnxruntime 的）
FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin",
    "1_Pooling/config.json",
]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def fetch(rel: str, out: Path) -> bool:
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{REPO}/resolve/main/{rel}"
    tmp = dst.with_suffix(dst.suffix + ".part")

    # 已有且非空则跳过
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [跳过] {rel:<36} {human(dst.stat().st_size)}")
        return True

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            print(f"  [下载] {rel:<36} {human(total)}")
            got = 0
            t0 = time.time()
            last = t0
            with tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 512)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last >= 5:          # 每 5 秒报一次
                        sp = got / (now - t0)
                        eta = (total - got) / sp if sp > 0 else 0
                        pct = got / total * 100 if total else 0
                        print(f"         {pct:5.1f}%  {human(got):>9} / "
                              f"{human(total):<9}  {human(sp)}/s  "
                              f"ETA {eta/60:.1f}min", flush=True)
                        last = now
        dt = time.time() - t0
        tmp.rename(dst)
        sp = got / dt if dt > 0 else 0
        print(f"  [完成] {rel:<36} {human(got)}  平均 {human(sp)}/s  "
              f"{dt/60:.1f}min")
        return True
    except Exception as e:
        print(f"  [失败] {rel}: {type(e).__name__}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 bge-m3")
    ap.add_argument("--out", type=Path, default=Path(r"F:\bge-m3"))
    args = ap.parse_args()

    print("=" * 72)
    print(f"下载 {REPO}")
    print("=" * 72)
    print(f"  镜像 {MIRROR}")
    print(f"  目标 {args.out}")

    # 先测速（下载一个小文件）
    print("\n  先测速 …")
    probe = "config.json"
    t0 = time.time()
    try:
        req = urllib.request.Request(f"{MIRROR}/{REPO}/resolve/main/{probe}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        dt = max(time.time() - t0, 1e-6)
        print(f"    下载 {len(data)} 字节用时 {dt:.1f}s "
              f"-> {human(len(data)/dt)}/s")
        if len(data) / dt < 50 * 1024:
            print("    ⚠️ 速度很慢（<50 KB/s），2.27 GB 主文件可能要 "
                  f"{2.27*1024**3/(len(data)/dt)/3600:.1f} 小时")
    except Exception as e:
        print(f"    测速失败: {e}")
        return 1

    print("\n  开始下载 …")
    t_all = time.time()
    ok = 0
    for rel in FILES:
        if fetch(rel, args.out):
            ok += 1

    print("\n" + "=" * 72)
    print(f"完成 {ok}/{len(FILES)}  总用时 {(time.time()-t_all)/60:.1f} 分钟")
    tot = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"  目录合计 {human(tot)}")
    print("=" * 72)
    return 0 if ok == len(FILES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
