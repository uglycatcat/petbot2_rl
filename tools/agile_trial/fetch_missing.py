#!/usr/bin/env python3
"""补下 parallel_pull.py 因镜像站卡住而缺失的分块（短超时 + 多次重试）。

用法：python3 fetch_missing.py [并发数]（默认 6），补完后重跑 parallel_pull.py 续传即可（它会组装 tar + 校验 sha256）。
"""
import concurrent.futures as cf
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pp", os.path.join(HERE, "parallel_pull.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

REG = os.environ.get("MIRROR_REGISTRY", "nvcr.m.daocloud.io")
REPO = os.environ.get("MIRROR_REPO", "nvidia/isaac-lab")
TAG = os.environ.get("MIRROR_TAG", "3.0.0-beta2")
OUT = os.path.join(HERE, "_images")
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAXTIME = os.environ.get("CHUNK_MAXTIME", "120")

token = pp.get_token(REG, REPO)
manifest = json.loads(pp.api(REG, f"/v2/{REPO}/manifests/{TAG}", token, accept=pp.ACCEPT))
if "manifests" in manifest:
    entry = next(e for e in manifest["manifests"]
                 if e.get("platform", {}).get("architecture") == "amd64")
    manifest = json.loads(pp.api(REG, f"/v2/{REPO}/manifests/{entry['digest']}", token, accept=pp.ACCEPT))

todo = []
for layer in manifest["layers"]:
    digest, size = layer["digest"], layer["size"]
    name = digest.split(":")[-1] + ".tar.gz"
    final = os.path.join(OUT, name)
    if os.path.exists(final):
        continue
    parts_dir = final + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    for idx, a in enumerate(range(0, size, pp.CHUNK)):
        b = min(size, a + pp.CHUNK) - 1
        part = os.path.join(parts_dir, f"{idx:05d}")
        if os.path.exists(part) and os.path.getsize(part) == b - a + 1:
            continue
        todo.append((name, digest, a, b, part))

print(f"待补分块 {len(todo)} 个（共 {len(manifest['layers'])} 层）", flush=True)
if not todo:
    print("无缺失，直接重跑 parallel_pull.py 组装即可", flush=True)
    sys.exit(0)


def fetch(item):
    name, digest, a, b, part = item
    want = b - a + 1
    for attempt in range(1, 31):
        cmd = ["curl", "-sL", "--fail", "--retry", "2", "--retry-delay", "2",
               "--max-time", MAXTIME, "-H", f"Authorization: Bearer {token}",
               "-r", f"{a}-{b}", "-o", part, f"https://{REG}/v2/{REPO}/blobs/{digest}"]
        if subprocess.run(cmd).returncode == 0 and os.path.exists(part) and os.path.getsize(part) == want:
            print(f"  OK   {name[:12]}… 分块 {os.path.basename(part)}（第 {attempt} 次）", flush=True)
            return True
    print(f"  FAIL {name[:12]}… 分块 {os.path.basename(part)}", flush=True)
    return False


with cf.ThreadPoolExecutor(JOBS) as ex:
    results = list(ex.map(fetch, todo))
print(f"成功 {sum(results)}/{len(results)}；现在重跑 parallel_pull.py 续传组装", flush=True)
sys.exit(0 if all(results) else 1)
