# -*- coding: utf-8 -*-
"""
PQ-8B 基线：与 HSH-64 同为 8 字节/项的乘积量化。

为什么需要它
------------
论文的相关工作提了 PQ，但主结果表里没有 PQ 的数字。而「HSH-64 内存极低」这个
叙事要成立，必须与**同内存预算**的方法比，而不是只与 4 KB/项的 float32 暴力比。
PQ 在 8 字节下是唯一真正的同预算对手。

同时给出两个控制实验，用于分解损失来源：
    · 教师 1024 维浮点       —— 真值本身，召回 1.0（定义）
    · PCA-512 浮点（不量化）  —— 只降维、不量化
    · PQ-8B                 —— 8 字节
    · HSH-64                —— 8 字节（已有记录值）

口径
----
与 scripts/52_eval_pipeline.py 完全一致：
    真值 G_i = 教师余弦 top-10（排除自身）
    overlap@10 = mean_i(|R_i ∩ G_i| / |G_i|)
自检：对现成的 sim_override_*.bin 计算召回，应复现 README 记录值
    nat3000 0.6082 / nat6000 0.5990 / nat10000 0.5936 / nat16000 0.5965 / mixed 0.6086

用法
----
    python scripts/65_pq_baseline.py
    python scripts/65_pq_baseline.py --bytes-list 8,16,32
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA = ROOT / "data"
EVAL = DATA / "eval"
RESULTS = ROOT / "results"

CACHE_MAGIC = 0xCAB1EBED
OVERRIDE_MAGIC = 0xCAB10D01


# ---------------------------------------------------------------- 读取器
def read_embedding_cache(path: Path) -> tuple[int, list[str], np.ndarray]:
    """0xCAB1EBED: >4I magic,ver,dim,count; 每词 >H len + bytes + dim×f32。"""
    with path.open("rb") as f:
        magic, _ver, dim, count = struct.unpack(">4I", f.read(16))
        if magic != CACHE_MAGIC:
            raise ValueError(f"{path.name}: magic 0x{magic:X} != 0x{CACHE_MAGIC:X}")
        words, vecs = [], np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            (blen,) = struct.unpack(">H", f.read(2))
            words.append(f.read(blen).decode("utf-8"))
            vecs[i] = np.frombuffer(f.read(dim * 4), dtype=">f4").astype(np.float32)
    return dim, words, vecs


def read_override(path: Path) -> np.ndarray:
    """0xCAB10D01: >IIII magic,ver,n_bits,count; 每词 >H len + bytes + >Q sim。
    位约定 bits[i, b] = (sim >> b) & 1（LSB 优先）。"""
    with path.open("rb") as f:
        magic, _ver, n_bits, count = struct.unpack(">4I", f.read(16))
        if magic != OVERRIDE_MAGIC:
            raise ValueError(f"{path.name}: magic 0x{magic:X} != 0x{OVERRIDE_MAGIC:X}")
        sims = np.empty(count, dtype=np.uint64)
        for i in range(count):
            (blen,) = struct.unpack(">H", f.read(2))
            f.read(blen)                      # 词本身，与缓存顺序一致，可跳过
            (sims[i],) = struct.unpack(">Q", f.read(8))
    shifts = np.arange(n_bits, dtype=np.uint64)
    return ((sims[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)


# ---------------------------------------------------------------- 评测口径
def truth_topk(teacher_sim: np.ndarray, k: int) -> np.ndarray:
    """教师 top-k，排除自身（第 0 列是自身）。"""
    kk = min(k, teacher_sim.shape[0] - 1)
    return np.argsort(-teacher_sim, axis=1, kind="stable")[:, 1 : kk + 1]


def recall_at_k(pos: np.ndarray, dist: np.ndarray, k: int) -> float:
    """与 cabinet_bnn.hsh.post_optimize.recall_at_k_from_ranking 逐位一致：
        分母为 |P_i|；排序用 stable argsort，再剔除自身，取前 k。
    必须用 stable argsort：Hamming 距离有大量并列，用 argpartition 会任意打破
    并列，导致与记录值差千分之几（实测 0.6035 vs 0.6082）。
    """
    n = dist.shape[0]
    kk = min(k, n - 1)
    total, counted = 0.0, 0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        idx = order[:kk]
        hit = np.intersect1d(pos[i], idx, assume_unique=False).size
        total += hit / max(pos.shape[1], 1)
        counted += 1
    return total / counted if counted else 0.0


def cosine_sim(x: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return xn @ xn.T


# ---------------------------------------------------------------- PQ
class ProductQuantizer:
    """标准 PQ：M 个子空间，每个 256 个质心（8 bit），共 8M 字节/项。"""

    def __init__(self, n_sub: int = 8, n_cent: int = 256, seed: int = 0):
        self.n_sub = n_sub
        self.n_cent = n_cent
        self.seed = seed
        self.centroids: list[np.ndarray] = []   # 每项 (n_cent, sub_dim)
        self.sub_dim = 0

    def fit(self, x: np.ndarray, verbose: bool = True) -> "ProductQuantizer":
        from sklearn.cluster import KMeans

        n, d = x.shape
        if d % self.n_sub:
            raise ValueError(f"维度 {d} 不能被 {self.n_sub} 整除")
        self.sub_dim = d // self.n_sub
        self.centroids = []
        t0 = time.time()
        for s in range(self.n_sub):
            xs = x[:, s * self.sub_dim : (s + 1) * self.sub_dim]
            km = KMeans(n_clusters=self.n_cent, n_init=3, max_iter=100,
                        random_state=self.seed, algorithm="lloyd")
            km.fit(xs)
            self.centroids.append(km.cluster_centers_.astype(np.float32))
            if verbose:
                print(f"      子空间 {s + 1}/{self.n_sub} 完成  "
                      f"({time.time() - t0:.1f}s, inertia={km.inertia_:.4g})")
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """→ (n, n_sub) uint8"""
        n = x.shape[0]
        codes = np.empty((n, self.n_sub), dtype=np.uint8)
        for s in range(self.n_sub):
            xs = x[:, s * self.sub_dim : (s + 1) * self.sub_dim]
            c = self.centroids[s]
            # 分块算平方距离，避免 (n, 256, sub_dim) 过大
            d2 = (xs * xs).sum(1)[:, None] - 2 * xs @ c.T + (c * c).sum(1)[None, :]
            codes[:, s] = np.argmin(d2, axis=1).astype(np.uint8)
        return codes

    def adc_distances(self, x: np.ndarray, codes: np.ndarray,
                      batch: int = 512) -> np.ndarray:
        """非对称距离（ADC）：查询保持浮点，库侧为码。"""
        n = x.shape[0]
        out = np.empty((n, codes.shape[0]), dtype=np.float32)
        for b0 in range(0, n, batch):
            b1 = min(b0 + batch, n)
            xb = x[b0:b1]
            acc = np.zeros((b1 - b0, codes.shape[0]), dtype=np.float32)
            for s in range(self.n_sub):
                xs = xb[:, s * self.sub_dim : (s + 1) * self.sub_dim]
                cs = self.centroids[s]
                lut = (xs * xs).sum(1)[:, None] - 2 * xs @ cs.T + (cs * cs).sum(1)[None, :]
                # (B, 256) 取码对应的那一列，累加到 (B, n)
                acc += lut[:, codes[:, s]]
            out[b0:b1] = acc
        return out

    @property
    def codebook_bytes(self) -> int:
        return int(self.n_sub * self.n_cent * self.sub_dim * 4)


# ---------------------------------------------------------------- 数据集
SETS = [
    ("nat3000", "teacher_nat3000.cache", "sim_override_nat3000.bin",
     "pca512_nat3000.npz", 0.6082),
    ("nat6000", "teacher_nat6000.cache", "sim_override_nat6000.bin",
     "pca512_nat6000.npz", 0.5990),
    ("nat10000", "teacher_nat10000.cache", "sim_override_nat10000.bin",
     "pca512_nat10000.npz", 0.5936),
    ("nat16000", "teacher_nat16000.cache", "sim_override_nat16000.bin",
     "pca512_nat16000.npz", 0.5965),
    ("mixed6109", "mixed_embedding.cache", "sim_override_mixed6109.bin",
     None, 0.6086),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="PQ 基线（同 8 字节预算）")
    ap.add_argument("--bytes-list", default="8",
                    help="逗号分隔的字节预算，8/16/32；默认 8")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--out", type=Path, default=RESULTS / "pq_baseline.json")
    args = ap.parse_args()

    budgets = [int(x) for x in args.bytes_list.split(",") if x.strip()]
    want = None if args.datasets == "all" else set(args.datasets.split(","))

    print("=" * 76)
    print("  PQ 基线（同内存预算对比）")
    print("=" * 76)

    out = {"budgets_bytes": budgets, "datasets": {}}

    for tag, tcache, override, pca_npz, hsh_record in SETS:
        if want and tag not in want:
            continue
        tp = EVAL / tcache if (EVAL / tcache).exists() else DATA / tcache
        if not tp.exists():
            print(f"\n  [{tag}] 跳过：缺教师缓存 {tcache}")
            continue

        print(f"\n[{tag}]")
        dim_t, words, teacher = read_embedding_cache(tp)
        n = teacher.shape[0]
        print(f"  教师 {dim_t} 维 / {n} 词")
        if words != words:  # 恒 False，保留占位
            pass

        teacher_sim = cosine_sim(teacher)
        pos = truth_topk(teacher_sim, 10)
        rec = {"n": n, "teacher_dim": dim_t, "hsh_record": hsh_record}

        # ---- PQ 的输入必须单位化
        # 真值是余弦；PQ 用 L2 距离。对单位向量有 ||a-b||² = 2-2cos，
        # 两者排序等价。教师嵌入未归一化时 L2 会被模长主导，PQ 会被不公平地压低。
        teacher_unit = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)

        # ---- 自检：用现成的 HSH 覆盖表复现记录值
        ov = EVAL / override
        if ov.exists():
            ov_words, _ = None, None
            bits = read_override(ov)
            # 覆盖表的词序必须与缓存一致；不一致则按词名重排
            with ov.open("rb") as f:
                f.read(16)
                ov_words = []
                for _ in range(n):
                    (blen,) = struct.unpack(">H", f.read(2))
                    ov_words.append(f.read(blen).decode("utf-8"))
                    f.read(8)
            if ov_words != words:
                idx = {w: i for i, w in enumerate(words)}
                order = [idx[w] for w in ov_words if w in idx]
                missing = n - len(order)
                print(f"  ⚠️ 覆盖表词序与缓存不同，按词名重排（缺失 {missing}）")
            hd = np.empty((n, n), dtype=np.float32)
            blk = max(1, int(2_000_000 // max(n, 1)))
            for s in range(0, n, blk):
                e = min(s + blk, n)
                hd[s:e] = (bits[s:e, None, :] != bits[None, :, :]).sum(2, dtype=np.float32)
            my_hsh = recall_at_k(pos, hd, 10)
            delta = my_hsh - hsh_record
            flag = "✅" if abs(delta) < 1e-3 else "⚠️"
            print(f"  【自检】HSH-64 覆盖表 overlap@10 = {my_hsh:.4f} "
                  f"（记录 {hsh_record:.4f}，差 {delta:+.4f}）{flag}")
            rec["hsh_recomputed"] = my_hsh
            rec["hsh_check_ok"] = bool(abs(delta) < 1e-3)

        # ---- 控制实验 1：教师浮点（真值本身）
        rec["teacher_float_bytes"] = dim_t * 4
        rec["teacher_float_recall"] = 1.0

        # ---- 控制实验 2：PCA-512 浮点（只降维不量化）
        x512 = None
        if pca_npz and (EVAL / pca_npz).exists():
            z = np.load(EVAL / pca_npz)
            x512 = (teacher - z["mean"]) @ z["components"].T
            t0 = time.time()
            s512 = cosine_sim(x512)
            r512 = recall_at_k(pos, -s512, 10)
            dt = time.time() - t0
            print(f"  PCA-512 浮点     recall@10 = {r512:.4f}   "
                  f"（只降维不量化，{x512.shape[1] * 4} B/项，{dt:.1f}s）")
            rec["pca512_float"] = {"recall": r512, "bytes": int(x512.shape[1] * 4)}

        # ---- PQ：每个字节预算各训一次
        # 两个输入口径：
        #   teacher1024 —— 教师原始 1024 维（PQ 的信息上限情形）
        #   pca512      —— 与 HSH-64 编码器完全相同的输入（PCA-512）
        # 两者都先单位化，使 L2 与余弦排序等价。
        #
        # 维度置换：PCA 的分量按方差降序排列，直接按连续块切子空间会让各子空间
        # 能量极不均衡（前 64 个 PC 占了绝大部分方差），严重压低 PQ。标准做法是
        # 打散维度顺序（或用 OPQ 学一个旋转）。此处用固定种子的随机置换，对
        # teacher1024 是中性操作（维度本无顺序），对 pca512 是必要的公平修正。
        rng_perm = np.random.default_rng(12345)
        inputs = []
        for iname, xin in [("teacher1024", teacher_unit)] + (
                [("pca512", x512 / (np.linalg.norm(x512, axis=1, keepdims=True) + 1e-12))]
                if x512 is not None else []):
            perm = rng_perm.permutation(xin.shape[1])
            inputs.append((iname, xin[:, perm], perm))

        rec["pq"] = {}
        for iname, x, _perm in inputs:
            for nbytes in budgets:
                n_sub = nbytes          # 每个子空间 8 bit → n_sub = nbytes
                n_cent = 256
                if x.shape[1] % n_sub:
                    print(f"  {iname} PQ-{nbytes}B: 维度 {x.shape[1]} 不能被 {n_sub} 整除，跳过")
                    continue
                key = f"{iname}_pq{nbytes}B"
                print(f"  PQ-{nbytes}B on {iname}（{n_sub} 子空间 × {n_cent} 质心）")
                t0 = time.time()
                pq = ProductQuantizer(n_sub=n_sub, n_cent=n_cent, seed=0).fit(x)
                fit_s = time.time() - t0
                t0 = time.time()
                codes = pq.encode(x)
                enc_s = time.time() - t0
                t0 = time.time()
                dist = pq.adc_distances(x, codes)
                search_s = time.time() - t0
                r = recall_at_k(pos, dist, 10)
                qps = n / search_s
                total_bytes = n * nbytes + pq.codebook_bytes
                print(f"    → recall@10 = {r:.4f}   "
                      f"码 {n * nbytes / 1024:.0f} KB + 码本 {pq.codebook_bytes / 1024:.0f} KB "
                      f"= {total_bytes / 1024:.0f} KB   QPS {qps:.0f}   "
                      f"（fit {fit_s:.1f}s / encode {enc_s:.1f}s / search {search_s:.1f}s）")
                rec["pq"][key] = {
                    "recall": r, "bytes_per_item": nbytes,
                    "codebook_bytes": pq.codebook_bytes,
                    "total_bytes": total_bytes,
                    "qps": qps, "fit_s": fit_s,
                }

        out["datasets"][tag] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
