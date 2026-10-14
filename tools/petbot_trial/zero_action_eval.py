"""零动作基线：用同一评估初态库跑"什么都不做"，检验 Recovery/success_rate 指标本身是否可信。

如果零动作也能拿到接近策略的成功率，说明判据（1 s 保持 + 四轮支撑 + 默认构型）可以被"原地不动"满足，
评估数字就不能当作恢复能力。用法（容器内，cwd=某个 run 副本）：
    /isaac-sim/python.sh /workspace/demo_env/zero_action_eval.py --headless --num_envs 512 --episodes 512
"""

import argparse
import json
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=20261014)
parser.add_argument("--output", default="artifacts/zero_action_eval.json")
parser.add_argument(
    "--action",
    default="",
    help="恒定动作（逗号分隔，按契约顺序 head,left_ear,right_ear,left_frame,right_frame,left_rear_wheel,right_rear_wheel）；"
    "默认全 0。传 -0,-0.5,-0.5,0,0,0,0 即为“恒定保持默认构型”的开环基线。",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym

    import agile.rl_env.tasks.stand_up.petbot  # noqa: F401
    from agile.petbot.pre_learn import pre_learn
    from agile.rl_env.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import load_cfg_from_registry

    env_cfg = load_cfg_from_registry("PetBot-Recovery-AGILE-v0", "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry("PetBot-Recovery-AGILE-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = min(args.num_envs, args.episodes)
    env_cfg.seed = args.seed
    env_cfg.sim.device = "cuda:0"
    agent_cfg.device = "cuda:0"
    agent_cfg.seed = args.seed
    agent_cfg.petbot_collection.split = "eval"
    agent_cfg.petbot_collection.seed = args.seed
    agent_cfg.petbot_collection.num_states = max(args.episodes, 256)
    env = gym.make("PetBot-Recovery-AGILE-v0", cfg=env_cfg)
    pre_learn(env.unwrapped, "PetBot-Recovery-AGILE-v0", agent_cfg)
    env.unwrapped.petbot_episode_records = []
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    action_dim = env.unwrapped.action_manager.total_action_dim
    if args.action:
        values = [float(x) for x in args.action.split(",")]
        assert len(values) == action_dim, f"动作维度 {len(values)} != {action_dim}"
    else:
        values = [0.0] * action_dim
    fixed = torch.tensor(values, device=wrapped.device).repeat(wrapped.num_envs, 1)
    print(f"[baseline] 恒定动作 {values}", flush=True)
    obs = wrapped.get_observations()
    while len(env.unwrapped.petbot_episode_records) < args.episodes:
        with torch.no_grad():
            obs, _, _, _ = wrapped.step(fixed)
    records = env.unwrapped.petbot_episode_records[: args.episodes]
    summary = {}
    for label in ("all", "upright", "inverted", "side_or_tilted"):
        rows = records if label == "all" else [r for r in records if r["initial_class"] == label]
        summary[label] = {
            "count": len(rows),
            "success_rate": (sum(r["success"] for r in rows) / len(rows)) if rows else None,
            "stable_at_end_rate": (sum(r["stable_at_end"] for r in rows) / len(rows)) if rows else None,
        }
    print(json.dumps(summary, indent=2))
    Path(args.output).write_text(json.dumps({"summary": summary, "episodes": records}, indent=2) + "\n")
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
os._exit(0)
