# -*- coding: utf-8 -*-
"""
把打包好的模型总成文件上传到 GitHub Release。

为什么需要脚本而不是手动上传：
    4 个包合计约 200 MB，网页上传有 25 MB/文件 的软限制（大文件要用表单），
    而且手动上传后还要改 README 里的链接。用 API 一次搞定。

前置条件（二选一）：
    A. 设置环境变量 GITHUB_TOKEN（Personal Access Token，需要 repo 权限）
    B. 装 gh CLI 并 `gh auth login`，然后把 token 读出来：
         gh auth token

本机当前【两者都没有】，所以这个脚本是把上传这一步准备好，
等有凭据时一条命令完成。

用法：
    # 1) 只检查，不发请求
    python scripts/61_upload_release.py --dry-run

    # 2) 真正上传（会先创建 release，再逐个上传 asset）
    set GITHUB_TOKEN=ghp_xxx
    python scripts/61_upload_release.py --tag v2.0.0

    # 3) 已存在 release，只补传 asset
    python scripts/61_upload_release.py --tag v2.0.0 --skip-create
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OWNER = "Sauomore"
REPO = "Cabinet_hsh64"
API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"

RELEASE_BODY = """## 自然词表规模评估 + 索引修复

本版本新增**自然中文词表**上训练的模型，用于回应「原论文 3,109 词规模太小」的批评，
并修复了 MIH 索引的两个缺陷。

### 为什么需要自然词表

原 3,109 词表里 **65% 是人造合成词**（10 个修饰词 × 约 200 个基础词，
如 `大U盘` `大一月` `大三月`），自然基础词只有约 1,114 个。
这类词表的 top-10 近邻边界比自然语料**清晰 2.18 倍**（归一化间隙 0.0196 vs 0.0090），
因此其上的召回数字不宜直接当作「真实词表检索」能力。

### 结果

| 词表 | 规模 | 裸码 | + 后处理 |
|---|---|---|---|
| 合成（论文） | 3,109 | 0.4988 | **0.7426** |
| 自然 | 3,000 | 0.2463 | **0.6082** |
| 自然 | 6,000 | 0.2453 | 0.5990 |
| 自然 | 10,000 | 0.2466 | 0.5936 |
| 自然 | 16,000 | 0.2519 | 0.5965 |

**规模不损害召回**：词表扩大 5.3 倍，后处理召回 0.6082 → 0.5965。
而 16,000 词时每个查询的真实邻居里 **62.3% 是新增词**，100% 的查询都有新邻居 ——
任务确实变难了，而 52 位码仍然找得到。

两阶段系统在合成词表上复现到 **0.8890**（论文 0.8641）。

### 修复

1. **`segment_count = 1` 静默失效** —— 段值 52 位被截断进 u32 键，且无法枚举段内汉明球，
   历史上对所有 radius 都返回 0 候选。现在明确报错（MIH 数学上要求 m ≥ 2）。
2. **`seg=2` 的 QPS 只有 6.2** —— 旧实现遍历 `0..2^segment_bits`（26 位 = 6700 万次）。
   改为枚举汉明球 `C(bits,≤d)`，在球体积大于段字典时反向遍历字典。
   **seg=2 提速 210 倍（6.2 → 1304 QPS）**，召回不变。正确性由新增单测保障：
   球枚举与暴力枚举逐元素相等。
3. **`--radii` 默认值从 `0,2,4,6,8` 改为 `0,4,8,12,16,18,20,22`** ——
   52 位码的正样本距离约 9-10 位，旧默认会让使用者开箱得到「召回 0.13」，
   误以为索引坏了。
4. **`--deep-hash` 模式不再强制要求 `--pca` 文件**。

### 每个包的内容

| 文件 | 说明 |
|---|---|
| `vocab.txt` | 词表 |
| `student_pca512.cache` | 学生嵌入（512 维，模型输入） |
| `teacher_bge_large.cache` | 教师嵌入（1024 维，评测真值来源） |
| `deep_hash_52.bin` | DeepHash 模型（512 → 512 → 52 bit） |
| `sim_override_52.bin` | 后处理覆盖码 |
| `pca512_basis.npz` | PCA 基 |
| `eval_result.json` | 该模型的实测召回 |
| `MANIFEST.json` | sha256 校验 + 复现命令 |

复现（解压后在该目录执行）：

```bash
cargo run --release --example benchmark_pure_hsh64 -- \\
    --embedding student_pca512.cache \\
    --deep-hash deep_hash_52.bin \\
    --sim-override sim_override_52.bin \\
    --reranker teacher_bge_large.cache \\
    --top-k 10 --queries <N>
```

⚠️ 必须带 `--reranker`：benchmark 的「真值」默认取自 `--embedding` 指向的向量，
只给 student 会得到「学生空间自洽性」这个另一个指标（nat3000 实测 0.5947 vs 0.6082）。

详细分析见 [docs/scale_evaluation.md](https://github.com/Sauomore/Cabinet_hsh64/blob/main/docs/scale_evaluation.md)。
"""


def token() -> str | None:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t.strip()
    if shutil.which("gh"):
        try:
            r = subprocess.run(["gh", "auth", "token"], capture_output=True,
                               text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return None


def api(method: str, url: str, tok: str, data: dict | None = None,
        binary: Path | None = None) -> tuple[int, dict]:
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "hsh64-release",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif binary is not None:
        body = binary.read_bytes()
        headers["Content-Type"] = "application/octet-stream"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:400]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="上传 Release 资源")
    ap.add_argument("--tag", default="v2.0.0")
    ap.add_argument("--name", default="HSH-64 v2: Natural Vocabulary Models")
    ap.add_argument("--dist", type=Path, default=ROOT / "dist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-create", action="store_true",
                    help="release 已存在，只补传 asset")
    args = ap.parse_args()

    zips = sorted(args.dist.glob("*.zip"))
    print("=" * 74)
    print("上传 Release 资源")
    print("=" * 74)
    print(f"  仓库   {OWNER}/{REPO}")
    print(f"  tag    {args.tag}")
    print(f"  目录   {args.dist}")
    print()

    if not zips:
        print("  ❌ dist 下没有 .zip，请先运行 scripts/58_package_release.py")
        return 1

    total = sum(z.stat().st_size for z in zips)
    print(f"  待上传 {len(zips)} 个包，合计 {total/1024**2:.1f} MB:")
    for z in zips:
        print(f"    {z.name:<46} {z.stat().st_size/1024**2:>7.1f} MB")
    print()

    tok = token()
    if not tok:
        print("  ❌ 找不到凭据（GITHUB_TOKEN / GH_TOKEN / gh auth token 都没有）")
        print()
        print("  三条路，任选其一：")
        print()
        print("  A. 用 token（推荐，可一次传完）")
        print("     1) 打开 https://github.com/settings/tokens")
        print("        Fine-grained token，权限：Repository permissions → Contents: Read and write")
        print("     2) PowerShell 里执行：")
        print("        $env:GITHUB_TOKEN = 'github_pat_xxx'")
        print(f"        python scripts/61_upload_release.py --tag {args.tag}")
        print()
        print("  B. 装 gh CLI")
        print("     winget install --id GitHub.cli")
        print("     gh auth login")
        print(f"     python scripts/61_upload_release.py --tag {args.tag}")
        print()
        print("  C. 网页手动上传（4 个文件，拖拽即可）")
        print("     1) 打开 https://github.com/Sauomore/Cabinet_hsh64/releases/new")
        print(f"        Tag: {args.tag}    Title: {args.name}")
        print("     2) 把下面 4 个文件拖进 Assets 区域：")
        for z in zips:
            print(f"        {z}")
        print("     3) 正文粘贴 —release-body 里的内容")
        print(f"        python scripts/61_upload_release.py --print-body")
        return 2

    print(f"  ✅ 找到凭据（长度 {len(tok)}）")
    if args.dry_run:
        print("  --dry-run：不发请求")
        return 0

    # 1) 找或建 release
    rel = None
    if not args.skip_create:
        st, cur = api("GET", f"{API}/repos/{OWNER}/{REPO}/releases/tags/{args.tag}",
                      tok)
        if st == 200:
            rel = cur
            print(f"  release 已存在（id={rel['id']}），补传 asset")
        else:
            st, rel = api("POST", f"{API}/repos/{OWNER}/{REPO}/releases", tok,
                          data={"tag_name": args.tag, "name": args.name,
                                "body": RELEASE_BODY, "draft": False,
                                "prerelease": False})
            if st not in (200, 201):
                print(f"  ❌ 创建 release 失败 HTTP {st}: {rel}")
                return 1
            print(f"  ✅ 已创建 release id={rel['id']}")
    else:
        st, rel = api("GET", f"{API}/repos/{OWNER}/{REPO}/releases/tags/{args.tag}",
                      tok)
        if st != 200:
            print(f"  ❌ 找不到 release {args.tag}（HTTP {st}）")
            return 1

    existing = {a["name"] for a in rel.get("assets", [])}

    # 2) 逐个上传
    ok = 0
    for z in zips:
        if z.name in existing:
            print(f"  [跳过] {z.name}（已存在）")
            ok += 1
            continue
        url = (f"{UPLOAD}/repos/{OWNER}/{REPO}/releases/{rel['id']}"
               f"/assets?name={z.name}")
        t0 = time.time()
        st, resp = api("POST", url, tok, binary=z)
        dt = time.time() - t0
        if st in (200, 201):
            print(f"  ✅ {z.name}  {z.stat().st_size/1024**2:.1f} MB  "
                  f"{dt:.1f}s")
            ok += 1
        else:
            print(f"  ❌ {z.name} HTTP {st}: {str(resp)[:200]}")

    print()
    print("=" * 74)
    print(f"完成 {ok}/{len(zips)}")
    print(f"  https://github.com/{OWNER}/{REPO}/releases/tag/{args.tag}")
    return 0 if ok == len(zips) else 1


if __name__ == "__main__":
    # --print-body 便捷用法
    if "--print-body" in sys.argv:
        print(RELEASE_BODY)
        raise SystemExit(0)
    raise SystemExit(main())
