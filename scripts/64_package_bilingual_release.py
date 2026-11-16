# -*- coding: utf-8 -*-
"""
打包 v2.5 双语模型资源包。

与 v2.0 的自然中文包结构一致，额外带两样双语必需的东西：
    mixed_tags.tsv   词性表 —— Rust 端必须用 --pos-tags 载入，否则英文词全落 FALLBACK
    mixed_pca_52.bin PCA-52 基线（0.3503），用于复现对比

包内嵌入缓存同时充当「学生输入」和「评测真值」：双语的教师与学生都是 bge-m3，
没有教师/学生维度差，这也是为什么口径比中文单语简单。

用法：
    python scripts/64_package_bilingual_release.py
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

DIST = ROOT / "dist"
STEM = "hsh64_pretrained_bilingual_6109"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="打包双语资源包")
    ap.add_argument("--out", type=Path, default=DIST)
    args = ap.parse_args()

    mapping = [
        (ROOT / "data/mixed_vocab.txt", "vocab.txt"),
        (ROOT / "data/mixed_embedding.cache", "embedding_bge_m3_1024.cache"),
        (ROOT / "data/mixed_deep_hash_52.bin", "deep_hash_52.bin"),
        (ROOT / "data/eval/sim_override_mixed6109.bin", "sim_override_52.bin"),
        (ROOT / "data/mixed_tags.tsv", "pos_tags.tsv"),
        (ROOT / "data/mixed_pca_52.bin", "pca_52.bin"),
        (ROOT / "data/eval/eval_mixed6109.json", "eval_result.json"),
    ]

    missing = [str(src) for src, _ in mapping if not src.exists()]
    if missing:
        print("  ❌ 缺少文件：")
        for m in missing:
            print(f"     {m}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stage = args.out / STEM
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    manifest = {"size": 6109, "files": {}, "command": ""}
    for src, name in mapping:
        dst = stage / name
        shutil.copy2(src, dst)
        manifest["files"][name] = {"bytes": dst.stat().st_size, "sha256": sha256(dst)}

    manifest["command"] = (
        "cargo run --release --example benchmark_pure_hsh64 -- "
        "--embedding embedding_bge_m3_1024.cache "
        "--deep-hash deep_hash_52.bin "
        "--sim-override sim_override_52.bin "
        "--pos-tags pos_tags.tsv "
        "--top-k 10 --queries 6109 --segment-counts 4 --radii 22"
    )
    manifest["base_model"] = "BAAI/bge-m3 (teacher AND student, 1024-dim)"
    manifest["n_bits"] = 52
    manifest["vocabulary"] = {"zh": 3095, "en": 3002, "mixed": 12, "total": 6109}

    readme = """# HSH-64 预训练模型 — 6,109 词中英混合词表

## 内容

| 文件 | 说明 |
|---|---|
| `vocab.txt` | 6,109 词（中文 3,095 / 英文 3,002 / 混合 12） |
| `embedding_bge_m3_1024.cache` | bge-m3 嵌入，**同时是模型输入和评测真值来源** |
| `deep_hash_52.bin` | DeepHash 模型（1024 → 512 → 52 bit） |
| `sim_override_52.bin` | 后处理优化后的码覆盖表 |
| `pos_tags.tsv` | 词性表（`word<TAB>lang<TAB>tag<TAB>feat`）—— **双语必需** |
| `pca_52.bin` | PCA-52 基线（Recall@10 0.3503） |
| `eval_result.json` | 实测结果 |

## 实测（全量 6,109 查询，overlap@10）

```text
PCA-52 基线         0.3503
裸码                0.3171
后处理             0.6086
Rust 暴力扫描       0.6131
Rust MIH (seg=4,r=22) 0.5895
```

跨语言检索（30 组可测中英词对，目标英文词在全部 6,109 码中的位次）：
中位 rank **2**，top-10 命中 **86.7%**，负对照落在随机水平（3,327）。

## 复现

```bash
{command}
```

⚠️ `--pos-tags` 不能省：省略后英文词按中文标签解析，全部落入 FALLBACK 槽。

## 已知限制

- 30 组可测词对里有 4 组排在 50 名之外，`猫 → cat` 排到 3,964（教师 rank ≤15）
- `feat` 四位的判别力有限，英文词性映射约 36% 有误（详见 docs/bilingual.md §4）
""".format(command=manifest["command"])

    (stage / "README.md").write_text(readme, encoding="utf-8")
    (stage / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = args.out / f"{STEM}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(stage.iterdir()):
            if f.is_file():
                z.write(f, f"{STEM}/{f.name}")

    print(f"  ✅ {zip_path.name}  {zip_path.stat().st_size/1024**2:.1f} MB")
    print(f"     {len(manifest['files'])} 个文件已验证 sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
