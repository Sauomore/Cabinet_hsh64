#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练完成后自动后处理并评估 Deep Hash 模型。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODELS = [
    ("tests/data/deep_hash_v3_64_distill_3109_h256_s42.bin", "h256_s42"),
    ("tests/data/deep_hash_v3_64_distill_3109_h256_s2024.bin", "h256_s2024"),
    ("tests/data/deep_hash_v3_64_distill_3109_h512_s42.bin", "h512_s42"),
    ("tests/data/deep_hash_v3_64_distill_3109_h512_s2024.bin", "h512_s2024"),
]

PARAMS = [
    # (pos_k, neg_weight, max_iters, suffix)
    (10, 0.1, 30, "pk10_nw01_mi30"),
    (10, 0.2, 30, "pk10_nw02_mi30"),
    (15, 0.1, 30, "pk15_nw01_mi30"),
    (15, 0.2, 30, "pk15_nw02_mi30"),
    (20, 0.1, 30, "pk20_nw01_mi30"),
    (20, 0.2, 30, "pk20_nw02_mi30"),
]


def run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=False)


def main():
    emb = "tests/data/embedding_3109.cache"
    teacher = "tests/data/reranker_3109_large_aligned.cache"

    override_files: list[Path] = []

    for model_path, tag in MODELS:
        model = Path(model_path)
        if not model.exists():
            print(f"[skip] 模型不存在: {model}", file=sys.stderr)
            continue
        for pos_k, neg_weight, max_iters, suffix in PARAMS:
            out = Path(f"tests/data/sim_override_distill_3109_{tag}_{suffix}.bin")
            if out.exists():
                print(f"[skip] 已存在: {out}", file=sys.stderr)
            else:
                run([
                    sys.executable, "scripts/post_optimize_codes_recall_64.py",
                    "--embedding-cache", emb,
                    "--model-type", "deep_hash",
                    "--model-path", str(model),
                    "--teacher-cache", teacher,
                    "--pos-k", str(pos_k),
                    "--neg-weight", str(neg_weight),
                    "--max-iters", str(max_iters),
                    "-o", str(out),
                ])
            override_files.append(out)

    # 评估所有新 override
    print("\n[eval] 单模型评估:", file=sys.stderr)
    for ov in override_files:
        if not ov.exists():
            continue
        run([
            sys.executable, "scripts/debug_recall.py",
            emb, teacher, str(ov),
        ])

    # 尝试 Ensemble
    existing = [
        Path("tests/data/sim_override_3109_h512_s42_recall_mi100.bin"),
        Path("tests/data/sim_override_3109_h512_s42_recall_mi30.bin"),
        Path("tests/data/sim_override_recall_64_3109_h512_large_input_s42_i30.bin"),
        Path("tests/data/sim_override_recall_64_3109_h1024_large_input_s42_i30.bin"),
        Path("tests/data/sim_override_recall_64_3109_h2048_large_input_s42_i30.bin"),
    ]
    all_files = [p for p in existing + override_files if p.exists()]
    print("\n[eval] Ensemble 评估:", file=sys.stderr)
    run([
        sys.executable, "scripts/ensemble_rank_search_2.py",
        emb, teacher,
    ] + [str(p) for p in all_files])


if __name__ == "__main__":
    main()
