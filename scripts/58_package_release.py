# -*- coding: utf-8 -*-
"""
打包模型总成文件（供 GitHub Release 分发）。

每个包包含复现评测所需的全部产物：
    vocab.txt              词表
    teacher_*.cache        教师嵌入（bge-large，1024 维）—— 评测真值来源
    student_*_d512.cache   学生嵌入（PCA-512）—— 与模型输入维度一致
    deep_hash_*.bin        DeepHash 模型（52 bit）
    sim_override_*.bin     后处理覆盖码
    MANIFEST.json          校验和 + 复现命令

为什么同时放 teacher 和 student：
    student 是模型的真实输入（512 维），必需。
    teacher 是评测真值的来源；不放它，使用者要自己下载 1.2 GB 的 bge-large
    才能重算 top-K，复现门槛会高很多。v1 的包也是这么做的。

用法：
    python scripts/58_package_release.py
    python scripts/58_package_release.py --sizes 3000 10000 16000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EVAL = ROOT / "data/eval"
DIST = ROOT / "dist"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(size: int) -> list[tuple[Path, str]] | None:
    """收集某规模的文件，返回 [(本地路径, 包内名字)]。"""
    tag = f"nat{size}"
    items: list[tuple[Path, str]] = []

    vocab = ROOT / f"data/natural_vocab_{size}.txt"
    if not vocab.exists():
        return None
    items.append((vocab, "vocab.txt"))

    mapping = [
        (EVAL / f"teacher_{tag}.cache", "teacher_bge_large.cache"),
        (EVAL / f"student_{tag}_d512.cache", "student_pca512.cache"),
        (EVAL / f"deep_hash_{tag}.bin", "deep_hash_52.bin"),
        (EVAL / f"sim_override_{tag}.bin", "sim_override_52.bin"),
        (EVAL / f"pca512_{tag}.npz", "pca512_basis.npz"),
        (EVAL / f"eval_{tag}.json", "eval_result.json"),
    ]
    for src, name in mapping:
        if src.exists():
            items.append((src, name))
    return items


def build_one(size: int) -> Path | None:
    items = collect(size)
    if not items:
        print(f"  [{size:>6,}] 缺少词表，跳过")
        return None

    stem = f"hsh64_pretrained_natural_{size}"
    stage = DIST / stem
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    manifest = {"size": size, "files": {}, "command": ""}
    for src, name in items:
        dst = stage / name
        shutil.copy2(src, dst)
        manifest["files"][name] = {
            "bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }

    # 复现命令
    #
    # ⚠️ 必须带 --reranker：benchmark 的「真值」默认取自 --embedding 指向的向量。
    #    若只给 student，真值就成了「学生空间的自洽性」，得到的是另一个指标
    #    （实测 nat3000：学生真值 0.5947 vs 教师真值 0.6082）。
    #    包里放 teacher 缓存就是为了让它当 reranker / 真值来源。
    manifest["command"] = (
        "cargo run --release --example benchmark_pure_hsh64 -- "
        "--embedding student_pca512.cache "
        "--deep-hash deep_hash_52.bin "
        "--sim-override sim_override_52.bin "
        "--reranker teacher_bge_large.cache "
        f"--top-k 10 --queries {size}"
    )
    manifest["base_model"] = "BAAI/bge-large-zh-v1.5 (teacher, 1024-dim)"
    manifest["student_dim"] = 512
    manifest["n_bits"] = 52

    readme = f"""# HSH-64 预训练模型 — {size:,} 词自然中文词表

## 内容

| 文件 | 说明 |
|---|---|
| `vocab.txt` | 词表（{size:,} 词，按词频排序） |
| `student_pca512.cache` | 学生嵌入（512 维，**模型的实际输入**） |
| `teacher_bge_large.cache` | 教师嵌入（1024 维，评测真值来源） |
| `deep_hash_52.bin` | DeepHash 模型（512 → 512 → 52 bit，{size} 词训练） |
| `sim_override_52.bin` | 后处理优化后的码覆盖表 |
| `pca512_basis.npz` | PCA 基（教师 → 学生的投影） |
| `eval_result.json` | 该模型的实测召回 |

## 复现

```bash
{manifest["command"]}
```

**两种口径别混用：**

| 命令 | 真值来源 | nat3000 实测 |
|---|---|---|
| 上面这条（带 `--reranker`） | 教师 bge-large | 与 `eval_result.json` 一致 |
| 去掉 `--reranker` | 学生 PCA-512 | 0.5947（自洽性，偏高/偏低都可能） |

`eval_result.json` 里的数字是**教师真值**口径。

## 说明

- 词表来自 wordfreq 官方中文词频表，过滤了单字与功能词
- 与论文的 3,109 词表只有约 10% 重叠（那个是合成的「修饰词×基础词」组合）
- 教师模型是 bge-large-zh-v1.5；学生输入是它的 PCA-512 截断
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    (stage / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = DIST / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(stage.iterdir()):
            if f.is_file():
                z.write(f, f"{stem}/{f.name}")
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 Release 资源")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[3000, 6000, 10000, 16000])
    args = ap.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("打包模型总成文件")
    print("=" * 74)

    made = []
    for size in args.sizes:
        z = build_one(size)
        if z:
            made.append(z)
            print(f"  [{size:>6,}] {z.name}  {z.stat().st_size/1024**2:.1f} MB")

    print()
    print("=" * 74)
    print(f"完成 {len(made)} 个包 -> {DIST}")
    for z in made:
        print(f"  {z.name:<44} {z.stat().st_size/1024**2:>7.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
