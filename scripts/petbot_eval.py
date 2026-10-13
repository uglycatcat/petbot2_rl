"""Independent PetBot episode evaluation. Run only in the training container."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--seed", type=int, default=20261014)
parser.add_argument("--output", default="artifacts/petbot_eval.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.num_envs < 1:
    parser.error("episodes and num_envs must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch

import agile.rl_env.tasks.stand_up.petbot  # noqa: F401
from agile.petbot.core import SuccessThresholds
from agile.petbot.model import load_model
from agile.petbot.pre_learn import pre_learn
from agile.rl_env.rsl_rl import RslRlVecEnvWrapper, make_rsl_rl_runner


def main():
    checkpoint = Path(args.checkpoint).resolve()
    saved = json.loads((checkpoint.parent / "params/petbot.json").read_text())
    if saved["source_model"]["source_sha256"] != load_model()["source_sha256"]:
        raise ValueError("Checkpoint source model differs from current model")
    from isaaclab.utils.io import load_yaml
    from isaaclab_tasks.utils import load_cfg_from_registry

    # Restore the training configuration rather than evaluating with current default hyperparameters.
    env_cfg = load_cfg_from_registry("PetBot-Recovery-AGILE-v0", "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry("PetBot-Recovery-AGILE-v0", "rsl_rl_cfg_entry_point")
    env_cfg.from_dict(load_yaml(str(checkpoint.parent / "params/env.yaml")))
    agent_cfg.from_dict(load_yaml(str(checkpoint.parent / "params/agent.yaml")))
    env_cfg.scene.num_envs = min(args.num_envs, args.episodes)
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.success = SuccessThresholds(**saved["success_thresholds"])
    # Saved USD absolute paths may point at another workspace; use the current prepared asset.
    from agile.petbot.asset_cfg import robot_cfg

    env_cfg.scene.robot.spawn.usd_path = robot_cfg().spawn.usd_path
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.seed = args.seed
    agent_cfg.petbot_collection.split = "eval"
    if args.seed == saved["collection"]["collection"]["seed"]:
        raise ValueError("Evaluation collection seed must differ from training collection seed")
    agent_cfg.petbot_collection.seed = args.seed
    agent_cfg.petbot_collection.num_states = max(args.episodes, 256)
    env = gym.make("PetBot-Recovery-AGILE-v0", cfg=env_cfg)
    try:
        pre_learn(env.unwrapped, "PetBot-Recovery-AGILE-v0", agent_cfg)
        env.unwrapped.petbot_episode_records = []
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = make_rsl_rl_runner(wrapped, agent_cfg, log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
        policy = runner.get_inference_policy(device=wrapped.device)
        obs = wrapped.get_observations()
        while len(env.unwrapped.petbot_episode_records) < args.episodes:
            with torch.no_grad():
                obs, _, _, _ = wrapped.step(policy(obs))
        episodes = env.unwrapped.petbot_episode_records[: args.episodes]
        summary = {}
        for label in ("all", "upright", "inverted", "side_or_tilted"):
            rows = episodes if label == "all" else [row for row in episodes if row["initial_class"] == label]
            summary[label] = {
                "count": len(rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
            }
        report = {
            "checkpoint": checkpoint.name,
            "evaluation": env.unwrapped.petbot_provenance,
            "summary": summary,
            "episodes": episodes,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
