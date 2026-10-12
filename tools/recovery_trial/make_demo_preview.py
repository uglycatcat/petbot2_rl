#!/usr/bin/env python3
"""把 Go2-W demo 视频抽帧拼成一张预览图，并打印视频信息。"""
import imageio
import imageio.v2 as iio
import numpy as np

BASE = "/workspace/variants/g1_go2w_paper/logs"
for ep in range(3):
    r = iio.get_reader(f"{BASE}/go2w_policy_ep{ep}.mp4")
    n = r.count_frames()
    meta = r.get_meta_data()
    print(f"ep{ep}: frames={n} size={meta.get('size')} fps={meta.get('fps')}")
    if ep == 1:
        idx = (0, n // 3, 2 * n // 3, n - 1)
        frames = [r.get_data(i) for i in idx]
        top = np.concatenate(frames[:2], axis=1)
        bot = np.concatenate(frames[2:], axis=1)
        imageio.imwrite(f"{BASE}/go2w_demo_preview.png", np.concatenate([top, bot], axis=0))
        print("预览图已写:", f"{BASE}/go2w_demo_preview.png")
