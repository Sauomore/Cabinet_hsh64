# -*- coding: utf-8 -*-
"""
把 PQ 对比搬到合成 3,109 词基准上，并补 OPQ（学习旋转）。

为什么要搬
----------
scripts/65 的 PQ 对比跑在**自然词表**（nat3000/nat16000），而编码器对比
（LSH/PCA/ITQ/DeepHash，scripts/67+70+72）跑在**合成 3,109**。
两张表不同源，无法直接回答「同 8 字节下 HSH 与 PQ 谁好」。本脚本在同一基准上重测。

口径说明（重要）
----------------
PQ 的码是**子码本下标**，不是比特，因此**无法使用本文的召回导向后处理**
（那是对比特做贪心翻转）。所以本脚本给出两个对照：
    · PQ 侧只有 ADC 距离（无后处理）
    · HSH 侧同时给裸码与后处理
结论必须同时呈现，不能只挑有利的一侧。

输入约定
--------
与 HSH 编码器一致：教师嵌入 → PCA-512（HSH 学生的实际输入）→ 单位化，
使 L2 与余弦排序等价。另有 1024 维原始输入变体作为 PQ 的信息上限情形。
维度置换启用（PCA 分量按方差排序，按连续块切分会让各子空间能量失衡）。

用法
----
    python scripts/74_pq_on_synth.py --teacher F:/Cabinet-BNN/results/bge_large_3109.cache
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Cabinet-BNN"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS = ROOT / "results"
CACHE_MAGIC = 0xCAB1EBED


def read_cache(path: Path):
    with path.open("rb") as f:
        magic, _v, dim, count = struct.unpack(">4I", f.read(16))
        if magic != CACHE_MAGIC:
            raise ValueError(f"magic 0x{magic:X}")
        words, vecs = [], np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            (blen,) = struct.unpack(">H", f.read(2))
            words.append(f.read(blen).decode("utf-8"))
            vecs[i] = np.frombuffer(f.read(dim * 4), dtype=">f4").astype(np.float32)
    return dim, words, vecs


def recall_with_dist(pos, dist, k):
    n = dist.shape[0]
    kk = min(k, n - 1)
    tot = 0.0
    for i in range(n):
        order = np.argsort(dist[i], kind="stable")
        order = order[order != i]
        tot += np.intersect1d(pos[i], order[:kk], assume_unique=False).size / pos.shape[1]
    return tot / n


class PQ:
    """标准乘积量化：n_sub 个子空间，每个 n_cent 个质心。"""

    def __init__(self, n_sub: int, n_cent: int = 256, seed: int = 0):
        self.n_sub, self.n_cent, self.seed = n_sub, n_cent, seed
        self.centroids: list[np.ndarray] = []
        self.sub_dim = 0

    def fit(self, x: np.ndarray) -> "PQ":
        from sklearn.cluster import KMeans
        n, d = x.shape
        self.sub_dim = d // self.n_sub
        self.centroids = []
        for s in range(self.n_sub):
            xs = x[:, s * self.sub_dim:(s + 1) * self.sub_dim]
            km = KMeans(n_clusters=self.n_cent, n_init=2, max_iter=60,
                        random_state=self.seed, algorithm="lloyd").fit(xs)
            self.centroids.append(km.cluster_centers_.astype(np.float64))
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        codes = np.empty((n, self.n_sub), dtype=np.uint8)
        for s in range(self.n_sub):
            xs = x[:, s * self.sub_dim:(s + 1) * self.sub_dim]
            c = self.centroids[s]
            d2 = (xs * xs).sum(1)[:, None] - 2 * xs @ c.T + (c * c).sum(1)[None, :]
            codes[:, s] = np.argmin(d2, 1).astype(np.uint8)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [self.centroids[s][codes[:, s]] for s in range(self.n_sub)], axis=1)

    def adc(self, x: np.ndarray, codes: np.ndarray, batch: int = 512) -> np.ndarray:
        n = x.shape[0]
        out = np.empty((n, codes.shape[0]), dtype=np.float32)
        for b0 in range(0, n, batch):
            b1 = min(b0 + batch, n)
            xb = x[b0:b1]
            acc = np.zeros((b1 - b0, codes.shape[0]), dtype=np.float32)
            for s in range(self.n_sub):
                xs = xb[:, s * self.sub_dim:(s + 1) * self.sub_dim]
                c = self.centroids[s]
                lut = (xs * xs).sum(1)[:, None] - 2 * xs @ c.T + (c * c).sum(1)[None, :]
                acc += lut[:, codes[:, s]]
            out[b0:b1] = acc
        return out

    @property
    def codebook_bytes(self):
        return int(self.n_sub * self.n_cent * self.sub_dim * 4)


def opq(x: np.ndarray, n_sub: int, n_cent: int = 256, n_iter: int = 15,
        seed: int = 0, verbose: bool = True):
    """OPQ：交替优化旋转 R 与 PQ 码本。返回 (pq, R, 量化误差)."""
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    pq = None
    err = float("inf")
    for it in range(n_iter):
        pq = PQ(n_sub, n_cent, seed=seed).fit(x @ R)
        codes = pq.encode(x @ R)
        Xhat = pq.decode(codes)
        err = float(np.mean((x @ R - Xhat) ** 2))
        if verbose:
            print(f"      OPQ iter {it:>2}  量化误差 {err:.6g}")
        # Procrustes: min ||X R - Xhat||_F  s.t. RᵀR = I
        U, _S, Vt = np.linalg.svd(x.T @ Xhat)
        R = U @ Vt
    pq = PQ(n_sub, n_cent, seed=seed).fit(x @ R)
    return pq, R, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--tag", default="synth3109")
    ap.add_argument("--bytes-list", default="8,16,32")
    ap.add_argument("--student-dim", type=int, default=512)
    ap.add_argument("--opq-bytes", default="8", help="对哪些字节预算额外跑 OPQ")
    ap.add_argument("--out", type=Path, default=RESULTS / "pq_on_synth.json")
    args = ap.parse_args()

    budgets = [int(x) for x in args.bytes_list.split(",") if x.strip()]
    opq_budgets = [int(x) for x in args.opq_bytes.split(",") if x.strip()]

    dim, words, teacher = read_cache(args.teacher)
    n = teacher.shape[0]
    tn = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-12)
    sim = tn @ tn.T
    pos = np.argsort(-sim, axis=1, kind="stable")[:, 1:11]
    print("=" * 88)
    print(f"  PQ / OPQ 在合成基准上 —— {args.tag}  教师 {dim}d / {n} 词")
    print("=" * 88)

    from sklearn.decomposition import PCA
    x512 = PCA(n_components=args.student_dim, random_state=0).fit_transform(tn.astype(np.float64))
    x512 = x512 / (np.linalg.norm(x512, axis=1, keepdims=True) + 1e-12)

    out = {"tag": args.tag, "n": n, "teacher_dim": dim,
           "student_dim": args.student_dim, "results": {}}
    t_all = time.time()
    rng_perm = np.random.default_rng(12345)

    for iname, x in (("pca512", x512), ("teacher1024", tn.astype(np.float64))):
        perm = rng_perm.permutation(x.shape[1])
        xp = x[:, perm]
        for nb in budgets:
            if xp.shape[1] % nb:
                print(f"  {iname} PQ-{nb}B: 维度不整除，跳过")
                continue
            key = f"{iname}_pq{nb}B"
            t0 = time.time()
            pq = PQ(nb, 256, seed=0).fit(xp)
            codes = pq.encode(xp)
            dist = pq.adc(xp, codes)
            r = recall_with_dist(pos, dist, 10)
            out["results"][key] = {
                "recall": r, "codebook_bytes": pq.codebook_bytes,
                "total_bytes": n * nb + pq.codebook_bytes, "fit_s": time.time() - t0,
            }
            print(f"  {key:<22} recall@10 {r:.4f}   码本 {pq.codebook_bytes/1024:.0f} KB"
                  f"   总 {(n * nb + pq.codebook_bytes)/1024:.0f} KB   ({time.time()-t0:.0f}s)")

            if nb in opq_budgets:
                t0 = time.time()
                pq_o, R_o, err = opq(xp, nb, 256, n_iter=15, seed=0, verbose=False)
                codes_o = pq_o.encode(xp @ R_o)
                dist_o = pq_o.adc(xp @ R_o, codes_o)
                r_o = recall_with_dist(pos, dist_o, 10)
                out["results"][f"{iname}_opq{nb}B"] = {
                    "recall": r_o, "codebook_bytes": pq_o.codebook_bytes,
                    "total_bytes": n * nb + pq_o.codebook_bytes,
                    "fit_s": time.time() - t0, "quant_err": err,
                }
                print(f"  {iname}_opq{nb}B        recall@10 {r_o:.4f}"
                      f"   ({time.time()-t0:.0f}s)   OPQ−PQ {r_o - r:+.4f}")

    # ---------- 与 HSH 各编码器对比
    sw_path = RESULTS / "bitwidth_sweep.json"
    lsh_path = RESULTS / "lsh_itq_seeds.json"
    if sw_path.exists():
        sw = json.load(io.open(sw_path, encoding="utf-8"))
        if sw.get("n") == n:
            print("\n" + "=" * 88)
            print("  同 8 字节对比（PQ 无后处理；HSH 给裸码与后处理两侧）")
            print("=" * 88)
            k52 = "52"
            # HSH 后处理最优（ITQ）来自 lsh_itq_seeds
            itq_post = None
            if lsh_path.exists():
                li = json.load(io.open(lsh_path, encoding="utf-8"))
                if li.get("n") == n:
                    itq_post = li["results"][k52]["itq_post"]["mean"]
            rows = []
            if "pca512_pq8B" in out["results"]:
                rows.append(("PQ-8B (PCA-512)", out["results"]["pca512_pq8B"]["recall"],
                             n * 8 + out["results"]["pca512_pq8B"]["codebook_bytes"]))
            if "pca512_opq8B" in out["results"]:
                rows.append(("OPQ-8B (PCA-512)", out["results"]["pca512_opq8B"]["recall"],
                             n * 8 + out["results"]["pca512_opq8B"]["codebook_bytes"]))
            if "teacher1024_pq8B" in out["results"]:
                rows.append(("PQ-8B (teacher 1024)",
                             out["results"]["teacher1024_pq8B"]["recall"],
                             n * 8 + out["results"]["teacher1024_pq8B"]["codebook_bytes"]))
            if itq_post is not None:
                rows.append(("HSH ITQ-52 + post", itq_post, n * 8))
            rows.append(("HSH PCA-52 裸码", sw["results"][k52]["pca_bare"], n * 8))
            print(f"    {'方法':<24}{'recall@10':>11}{'总字节':>12}{'后处理':>9}")
            for name, r, tb in rows:
                has_post = "post" in name
                print(f"    {name:<24}{r:>11.4f}{tb:>12,}{'是' if has_post else '—':>9}")
            out["comparison"] = [{"method": a, "recall": b, "total_bytes": int(c)}
                                 for a, b, c in rows]

    out["seconds"] = time.time() - t_all
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  总耗时 {out['seconds'] / 60:.1f} min   已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
