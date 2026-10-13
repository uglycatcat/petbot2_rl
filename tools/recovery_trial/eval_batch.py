#!/usr/bin/env python3
"""批量评估多个检查点（论文口径 C），并发受控，结束时汇总成表。

用法：
    python3 eval_batch.py paper_v1:model_2000.pt v2_exp_ori:model_9999.pt ...
    python3 eval_batch.py --jobs 4 <同上>

要点：
- 评估必须用训练时同一份代码 → 变体走 PYTHONPATH 指向自己的 src（paper_v1 用容器默认安装）。
- 多个 Isaac Sim 实例同卡并发会吃显存：默认每卡最多 2 个（--gpu-slots 可调）。
- 日志与结果都写在挂载目录内，宿主机可直接读。
"""
import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HOST_BASE = "/data/anna/ROS_About/reference_project"
GPU_POOL = [2, 3]


def spec(variant: str):
    """返回 (容器, 容器内根目录, 容器内 PYTHONPATH, 宿主机根目录)"""
    if variant == "paper_v1":
        return ("recovery-train", "/workspace/recovery", "",
                f"{HOST_BASE}/recovery")
    return ("recovery-vis", f"/workspace/variants/{variant}",
            f"/workspace/variants/{variant}/src",
            f"{HOST_BASE}/recovery_variants/{variant}")


def launch(variant: str, ckpt: str, gpu: int, host_tag: str):
    ct, root, pp, _ = spec(variant)
    tag = ckpt.replace(".pt", "")
    inner = (
        f"cd {root} && CUDA_VISIBLE_DEVICES={gpu} "
        + (f"PYTHONPATH={pp} " if pp else "")
        + "/isaac-sim/python.sh /workspace/recovery_env/eval_recovery.py "
        + f"--checkpoint $(ls {root}/logs/rsl_rl/thunder_recovery/*{variant}*/{ckpt} | head -1) "
        + f"--num_envs {os.environ.get('EVAL_ENVS', '512')} --episodes {os.environ.get('EVAL_EPS', '1')} {os.environ.get('EVAL_EXTRA', '')} --out {root}/logs/eval_{variant}_{tag}.json "
        + f"> logs/eval_{variant}_{tag}.out 2>&1"
    )
    subprocess.run(["docker", "exec", "-d", ct, "bash", "-lc", inner], check=False)
    return os.path.join(spec(variant)[3], "logs", f"eval_{variant}_{tag}.out")


def parse(path: str):
    """从评估日志里抽取关键行"""
    try:
        with open(path, errors="ignore") as fh:
            txt = fh.read()
    except OSError:
        return None
    if "评估结果" not in txt:
        return None
    res = {}

    def grab(pat, key, cast=str, flags=re.M):
        m = re.search(pat, txt, flags)
        if m:
            res[key] = cast(m.group(1))

    grab(r"C\) 论文判据\(高度0\.42\+腿\)\s*:\s*([\d.]+)%", "C", float)
    grab(r"高度\+姿态达标\(最后一步\)\s*:\s*([\d.]+)%", "stand", float)
    grab(r"^\s*h42\s+([\d.]+)%", "h42", float)
    # 末态分布是 p10 / p50 / p90 三列，这里取全部三个分位
    m = re.search(r"leg_joint_dev_rad\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    if m:
        res["jdev_p10"], res["jdev_p50"], res["jdev_p90"] = (float(x) for x in m.groups())
    m = re.search(r"leg_max_joint_vel_rad_s\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    if m:
        res["vmax_p10"], res["vmax_p50"], res["vmax_p90"] = (float(x) for x in m.groups())
    for name, zh in (("upright", "直立"), ("inverted", "倒置"), ("side", "侧躺"), ("tilted", "斜躺")):
        m = re.search(rf"{name}\s+样本\s+\d+\s+A[^%]*%[^%]*%[^%]*%\s+站姿\s+([\d.]+)%", txt)
        if m:
            res[name] = float(m.group(1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", help="变体:检查点，如 paper_v1:model_2000.pt")
    ap.add_argument("--jobs", type=int, default=2, help="每卡并发数（默认 2）")
    ap.add_argument("--parse-only", action="store_true",
                    help="不跑评估，只把磁盘上已有的 eval_*.out 重新汇总成表")
    args = ap.parse_args()

    if args.parse_only:
        import glob
        files = sorted(glob.glob(f"{HOST_BASE}/recovery/logs/eval_paper_v1_*.out")
                       + glob.glob(f"{HOST_BASE}/recovery_variants/*/logs/eval_*.out"))
        rows = []
        for f in files:
            r = parse(f)
            if not r:
                continue
            base = os.path.basename(f)[:-4]           # eval_<variant>_model_N
            try:
                v, ck = base[len("eval_"):].rsplit("_model_", 1)
                ck = f"model_{ck}"
            except ValueError:
                continue
            rows.append((v, ck, r))
        rows.sort(key=lambda x: (x[0], int(re.sub(r"\D", "", x[1]) or 0)))
        print(f"{'实验':<16}{'检查点':<15}{'C判据':<8}{'站姿':<8}{'h42':<8}"
              f"{'腿偏差p50':<11}{'腿速度p50':<11}分层(直立/倒置/侧躺/斜躺)")
        for v, ck, r in rows:
            f1 = lambda k, s="%": (f"{r[k]:.2f}{s}" if k in r else "—")
            print(f"{v:<16}{ck:<15}{f1('C'):<8}{f1('stand'):<8}{f1('h42'):<8}"
                  f"{r.get('jdev_p50', float('nan')):<11.3f}{r.get('vmax_p50', float('nan')):<11.3f}"
                  f"{f1('upright')}/{f1('inverted')}/{f1('side')}/{f1('tilted')}")
        return

    work = []
    for i, p in enumerate(args.pairs):
        v, ck = p.split(":")
        work.append((v, ck, GPU_POOL[i % len(GPU_POOL)]))

    print(f"共 {len(work)} 个评估，每卡并发 {args.jobs}\n", flush=True)
    sem_slots = args.jobs * len(GPU_POOL)

    def do(item):
        v, ck, gpu = item
        t0 = time.time()
        out = launch(v, ck, gpu, ck)
        while time.time() - t0 < 1800:
            r = parse(out)
            if r is not None:
                return v, ck, r, time.time() - t0
            time.sleep(15)
        return v, ck, None, time.time() - t0

    with ThreadPoolExecutor(max_workers=sem_slots) as ex:
        results = list(ex.map(do, work))

    print(f"{'变体':<14}{'检查点':<14}{'C(论文判据)':<12}{'站姿达标':<10}{'h42':<8}"
          f"{'腿偏差p50':<11}{'腿速度p50':<11}  分层(直立/倒置/侧躺/斜躺)")
    for v, ck, r, el in results:
        if not r:
            print(f"{v:<14}{ck:<14}{'评估失败':<12}（{el:.0f}s）")
            continue
        def g(k, d="—"):
            return f"{r[k]:.2f}" if k in r and isinstance(r[k], float) else (r.get(k, d) if k in r else d)
        print(f"{v:<14}{ck:<14}{g('C')+'%':<12}{g('stand')+'%':<10}{g('h42')+'%':<8}"
              f"{str(r.get('jdev_p50','—')):<11}{str(r.get('vmax_p50','—')):<11}  "
              f"{g('upright')}/{g('inverted')}/{g('side')}/{g('tilted')}")


if __name__ == "__main__":
    main()
