# -*- coding: utf-8 -*-
"""
网络下载工具（适配本机的 SteamTools 加速器环境）。

环境事实（已实测）：
    · hosts 文件把 github.com / raw.githubusercontent.com / huggingface.co
      等域名指向 127.0.0.1
    · 但本机跑着 SteamTools 加速器，监听 127.0.0.1:443 做 TLS 转发
      （证书签发者 CN=SteamTools Certificate, O=BeyondDimension）
    · 所以【网络是通的】—— urllib 直接请求 github 返回 HTTP 200
    · 真正的问题是 nltk.download() 有自己的 SSRF 防护，
      看到解析结果是 127.0.0.1 就拒绝，报 "SSRF attempt to restricted IP"

结论：
    不要试图"绕过投毒"（DoH 等），因为根本没被墙。
    直接自己下载 + 解压即可，然后把 NLTK_DATA 指向解压目录。

用法：
    python scripts/net_fetch.py --nltk            # 下载 nltk 标注数据
    python scripts/net_fetch.py --url <URL> --out <文件>
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

#: 需要绕过投毒的域名
POISONED = {
    "raw.githubusercontent.com",
    "github.com",
    "objects.githubusercontent.com",
    "codeload.github.com",
}

DOH_ENDPOINTS = [
    ("https://1.1.1.1/dns-query", "application/dns-json"),
    ("https://dns.google/resolve", "application/dns-json"),
]


def resolve_via_doh(host: str) -> list[str]:
    """用 DNS over HTTPS 解析域名，返回 IPv4 列表。"""
    for ep, _ in DOH_ENDPOINTS:
        try:
            url = f"{ep}?name={host}&type=A"
            req = urllib.request.Request(url, headers={"accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode())
            ips = [a["data"] for a in d.get("Answer", []) if a.get("type") == 1]
            if ips:
                print(f"    DoH({ep.split('/')[2]}) -> {', '.join(ips)}")
                return ips
        except Exception as e:
            print(f"    DoH {ep} 失败: {type(e).__name__}")
    return []


def fetch(url: str, out: Path | None = None, timeout: int = 120) -> bytes:
    """直接下载。本机 SteamTools 会转发，不需要特殊处理。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"    已写入 {out}  ({len(data)/1024:.0f} KB)")
    return data


#: nltk 需要的数据包
NLTK_PKGS = [
    ("taggers/averaged_perceptron_tagger_eng.zip", "taggers"),
    ("taggers/averaged_perceptron_tagger.zip", "taggers"),
]


def find_nltk_dir() -> Path:
    """找一个可写的 nltk_data 目录。"""
    for cand in [Path("F:/nltk_data"), Path.home() / "nltk_data",
                 Path("C:/nltk_data")]:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            t = cand / ".wtest"
            t.write_text("x")
            t.unlink()
            return cand
        except Exception:
            continue
    raise RuntimeError("找不到可写的 nltk_data 目录")


def fetch_nltk() -> int:
    base = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
    nd = find_nltk_dir()
    print("=" * 68)
    print("下载 nltk 数据")
    print("=" * 68)
    print(f"  目标 {nd}")

    ok = 0
    for rel, sub in NLTK_PKGS:
        url = f"{base}/{rel}"
        name = Path(rel).name
        print(f"\n  [{name}]")
        try:
            data = fetch(url)
            zpath = nd / name
            zpath.write_bytes(data)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(nd / sub)
            zpath.unlink()
            print(f"    ✅ 解压到 {nd / sub}")
            ok += 1
        except Exception as e:
            print(f"    ❌ {type(e).__name__}: {e}")

    print(f"\n  完成 {ok}/{len(NLTK_PKGS)}")
    print(f"  如 nltk 找不到，设置环境变量: NLTK_DATA={nd}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="绕过 DNS 投毒下载")
    ap.add_argument("--url")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--nltk", action="store_true")
    args = ap.parse_args()

    if args.nltk:
        return fetch_nltk()
    if args.url:
        fetch(args.url, args.out)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
