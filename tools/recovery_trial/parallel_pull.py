#!/usr/bin/env python3
"""多连接镜像拉取器：产出可直接 `docker load -i` 的 tarball。

为什么不用 docker pull / crane pull：
  * dockerd 配了 HTTP_PROXY=127.0.0.1:7890（该代理未运行），docker pull 全部失败；
  * crane pull 单连接实测 ~1 MB/s，而本机对 NGC 国内镜像站多连接聚合可达 ~13 MB/s。

产物格式与 crane 的 tarball 一致（已用 alpine 验证 docker load 可用）：
  <config-digest>            镜像配置 JSON
  <layer-digest-hex>.tar.gz  原始 gzip 层 blob
  manifest.json              [{Config, RepoTags, Layers}]

支持续传：分块落在 <输出目录>/<层>.parts/，重跑时补未完成的分块。
每个 blob 下载后校验 sha256（与 digest 比对），不一致直接失败。

用法：
  parallel_pull.py <registry>/<repo>:<tag> <输出 tar 路径> [-j 并发数]
例：
  parallel_pull.py nvcr.m.daocloud.io/nvidia/isaac-lab:2.3.2 ./isaac-lab-2.3.2.tar -j 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.parse
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])
CHUNK = 32 * 1024 * 1024          # 分块大小（续传/重试的最小粒度）


def log(*a):
    print(*a, flush=True)


def get_token(registry: str, repo: str) -> str:
    """匿名 token：先探 /v2/ 拿 WWW-Authenticate，再到 realm 换 token。"""
    req = urllib.request.Request(f"https://{registry}/v2/", headers={"User-Agent": "parallel-pull"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return ""
    except urllib.error.HTTPError as e:
        hdr = e.headers.get("WWW-Authenticate", "")
    # 形如：Bearer realm="https://...",service="...",scope="repository:x/y:pull"
    params = dict(re.findall(r'(\w+)="([^"]*)"', hdr))
    params.setdefault("scope", f"repository:{repo}:pull")
    url = params["realm"] + "?" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if k in ("service", "scope")})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "parallel-pull"}), timeout=30) as r:
        return json.load(r).get("token", "")


def api(registry: str, path: str, token: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "parallel-pull"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(f"https://{registry}{path}", headers=headers), timeout=60) as r:
        return r.read()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def download_blob(registry: str, repo: str, digest: str, size: int, url_path: str,
                  out_dir: str, out_name: str, token: str, jobs: int) -> str:
    """多连接分块下载一个 blob，校验 sha256，返回最终文件路径。

    分块固定为 CHUNK 大小（续传与重试的最小粒度），由 jobs 个连接并发取。
    """
    final = os.path.join(out_dir, out_name)
    if os.path.exists(final) and sha256(final) == digest.split(":")[-1]:
        log(f"  已存在且校验通过: {out_name}")
        return final
    parts_dir = os.path.join(out_dir, f"{out_name}.parts")
    os.makedirs(parts_dir, exist_ok=True)
    bounds = []
    for i, a in enumerate(range(0, size, CHUNK)):
        bounds.append((i, a, min(size, a + CHUNK) - 1))
    lock = threading.Lock()
    state = {"done": 0}
    url = f"https://{registry}{url_path}"

    def fetch(item):
        idx, a, b = item
        part = os.path.join(parts_dir, f"{idx:05d}")
        want = b - a + 1
        if os.path.exists(part) and os.path.getsize(part) == want:
            with lock:
                state["done"] += want
            return
        for attempt in (1, 2, 3, 4):
            cmd = ["curl", "-sL", "--fail", "--retry", "2", "--retry-delay", "3", "--max-time", "1800",
                   "-H", f"Authorization: Bearer {token}", "-r", f"{a}-{b}", "-o", part, url]
            if subprocess.run(cmd).returncode == 0 and os.path.exists(part) and os.path.getsize(part) == want:
                with lock:
                    state["done"] += want
                return
            log(f"  分块 {idx} 第 {attempt} 次失败，重试")
        raise RuntimeError(f"分块 {idx} 下载失败: {out_name}")

    with cf.ThreadPoolExecutor(max(1, min(jobs, len(bounds)))) as ex:
        list(ex.map(fetch, bounds))
    with open(final, "wb") as out:
        for idx, _, _ in bounds:
            with open(os.path.join(parts_dir, f"{idx:05d}"), "rb") as p:
                while True:
                    blk = p.read(1 << 22)
                    if not blk:
                        break
                    out.write(blk)
    got = sha256(final)
    if got != digest.split(":")[-1]:
        os.remove(final)
        raise RuntimeError(f"sha256 不匹配 {out_name}: {got} != {digest}")
    log(f"  完成 {out_name} ({size / 1e6:.1f} MB, {len(bounds)} 分块) sha256 校验通过")
    for idx, _, _ in bounds:
        os.remove(os.path.join(parts_dir, f"{idx:05d}"))
    os.rmdir(parts_dir)
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="<registry>/<repo>:<tag>")
    ap.add_argument("out", help="输出 tar 路径")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="并发连接数（默认 8）")
    ap.add_argument("--platform", default="linux/amd64")
    args = ap.parse_args()

    registry, rest = args.ref.split("/", 1)
    repo, tag = rest.rsplit(":", 1)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    log(f"[1/4] 鉴权 {registry}/{repo}:{tag}")
    token = get_token(registry, repo)

    log("[2/4] 取 manifest")
    raw = api(registry, f"/v2/{repo}/manifests/{tag}", token, accept=ACCEPT)
    m = json.loads(raw)
    if "manifests" in m:
        want = args.platform.split("/")
        picked = None
        for e in m["manifests"]:
            p = e.get("platform", {})
            if p.get("os") == want[0] and p.get("architecture") == want[1]:
                picked = e["digest"]
                break
        if picked is None:
            raise SystemExit(f"manifest 中没有 {args.platform}")
        raw = api(registry, f"/v2/{repo}/manifests/{picked}", token, accept=ACCEPT)
        m = json.loads(raw)
    layers = m["layers"]
    total = sum(l["size"] for l in layers)
    log(f"  配置 {m['config']['digest'][:19]}… 层 {len(layers)} 个，合计 {total / 1e9:.2f} GB(压缩)")

    log("[3/4] 多连接下载")
    cfg_digest = m["config"]["digest"]
    cfg_name = cfg_digest          # 与 crane 一致，用完整 digest 作为文件名
    cfg_path = os.path.join(out_dir, cfg_name)
    body = api(registry, f"/v2/{repo}/blobs/{cfg_digest}", token)
    with open(cfg_path, "wb") as f:
        f.write(body)
    layer_names = []
    # 下载按层大小从大到小（先啃大的），但 manifest.json 里的 Layers 必须保持
    # registry 的原始顺序：docker load 会按位置把它与配置里的 rootfs.diff_ids 对应。
    for i, l in enumerate(sorted(layers, key=lambda x: -x["size"])):
        name = l["digest"].split(":")[-1] + ".tar.gz"
        log(f"  层 {i + 1}/{len(layers)} {l['digest'][:19]}… {l['size'] / 1e6:.1f} MB")
        download_blob(registry, repo, l["digest"], l["size"],
                      f"/v2/{repo}/blobs/{l['digest']}", out_dir, name, token, args.jobs)
    layer_names = [l["digest"].split(":")[-1] + ".tar.gz" for l in layers]

    log("[4/4] 组装 tarball")
    manifest = [{"Config": cfg_name, "RepoTags": [args.ref], "Layers": layer_names}]
    man_path = os.path.join(out_dir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f)
    with tarfile.open(args.out, "w") as tar:
        tar.add(cfg_path, arcname=cfg_name)
        for n in dict.fromkeys(layer_names):   # 层 digest 可能重复，tar 里只存一份
            tar.add(os.path.join(out_dir, n), arcname=n)
        tar.add(man_path, arcname="manifest.json")
    for n in dict.fromkeys(layer_names):
        os.remove(os.path.join(out_dir, n))
    os.remove(cfg_path)
    os.remove(man_path)
    log(f"完成: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")
    log(f"装载: docker load -i {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
