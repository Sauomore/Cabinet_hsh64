# -*- coding: utf-8 -*-
"""
批量跑自然词表的规模曲线：6k / 10k / 16k。

3k 已经单独跑过（见 results/eval_runs.jsonl）。这里补齐其余三档。
每个规模都是：生成教师嵌入 → PCA-512 → 训练 → 后处理 → 评测。

之所以写成脚本而不是命令行串联：
    16k 的后处理是 O(N²)，耗时不确定；用脚本能记录每档的开始/结束时间，
    中途失败也能看到跑到哪了。

用法：
    python scripts/54_run_scale_curve.py
    python scripts/54_run_scale_curve.py --sizes 6000 10000 16000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LOG = ROOT / "results/scale_curve.log"


def echo(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(size: int, epochs: int, seed: int, embed_model: Path) -> int:
    vocab = ROOT / f"data/natural_vocab_{size}.txt"
    if not vocab.exists():
        echo(f"  ❌ 缺少词表 {vocab}")
        return 1

    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/52_eval_pipeline.py"),
        "--vocab", str(vocab),
        "--generate-embeddings",
        "--embed-model", str(embed_model),
        "--tag", f"nat{size}",
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--post-iters", "10",
        "--neg-weight", "1.0",
    ]
    echo(f"  启动 nat{size}: {' '.join(cmd[2:6])} …")
    t0 = time.time()
    logf = ROOT / f"results/eval_nat{size}.log"
    with logf.open("w", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            lf.write(line)
            s = line.rstrip()
            if any(k in s for k in ("[训练] 完成", "Recall@", "iter ",
                                    "唯一 sim", "[后处理] 用时", "结果 ->",
                                    "Error", "Traceback", "❌")):
                print("    " + s[:110], flush=True)
        p.wait()
    dt = (time.time() - t0) / 60
    echo(f"  nat{size} 结束 退出码={p.returncode} 用时 {dt:.1f} 分钟  log={logf.name}")
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="自然词表规模曲线")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6000, 10000, 16000])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed-model", type=Path,
                    default=Path(r"F:\hsh64_models\bge-large-zh-v1.5"))
    args = ap.parse_args()

    echo("=" * 70)
    echo(f"规模曲线：{args.sizes}")
    echo("=" * 70)

    results = []
    for size in args.sizes:
        rc = run_one(size, args.epochs, args.seed, args.embed_model)
        results.append((size, rc))

    echo("\n" + "=" * 70)
    echo("汇总")
    echo("=" * 70)
    for size, rc in results:
        echo(f"  nat{size:<6} {'OK' if rc == 0 else f'失败({rc})'}")
    return 0 if all(rc == 0 for _, rc in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
