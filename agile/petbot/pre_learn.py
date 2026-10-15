"""Called by AGILE train/play before the first policy rollout."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from .collection import collect, probe_standing, validate_dataset
from .core import reset_pool_indices
from .model import JOINT_ORDER, POSITION_JOINTS, PROJECT_ROOT, REAR_WHEELS, load_model, prepared_asset


def pre_learn(env, task_name, agent_cfg):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True).strip()

    # A run must reference a reproducible demo revision, including the first run.
    if git("status", "--porcelain"):
        raise RuntimeError("Commit the demo changes before starting a run; use a fixed, clean code checkout.")
    robot = env.scene["robot"]
    if set(robot.joint_names) != set(JOINT_ORDER) or env.action_manager.total_action_dim != 7:
        raise RuntimeError(f"Unexpected imported articulation/action layout: {robot.joint_names}")
    position = env.action_manager.get_term("joint_position")
    rear = env.action_manager.get_term("rear_velocity")
    if position._joint_names != list(POSITION_JOINTS) or rear._joint_names != list(REAR_WHEELS):
        raise RuntimeError("Resolved action order differs from PetBot control contract.")
    env.petbot_standing_height = probe_standing(env)
    cfg = agent_cfg.petbot_collection
    external = getattr(agent_cfg, "petbot_eval_dataset", None)
    if external:
        if cfg.split != "eval":
            raise ValueError("External evaluation states cannot be used for training")
        payload = torch.load(external, map_location="cpu", weights_only=True)
        metadata = dict(payload["metadata"])
        if metadata.get("quaternion_order") != "xyzw":
            if metadata.get("schema") != 1 or not getattr(agent_cfg, "petbot_allow_legacy_xyzw", False):
                raise ValueError("Dataset quaternion contract missing; explicitly opt in to known Lab 3 legacy xyzw data")
            metadata = {**metadata, "legacy_schema": 1, "schema": 2, "quaternion_order": "xyzw",
                        "root_velocity_frame": "world_com", "legacy_xyzw_explicitly_confirmed": True}
        import hashlib
        usd, _ = prepared_asset()
        if metadata["usd_sha256"] != hashlib.sha256(usd.read_bytes()).hexdigest():
            raise ValueError("External state library USD differs")
        if metadata["physics_dt"] != env.physics_dt or metadata["decimation"] != env.cfg.decimation:
            raise ValueError("External state library simulation timing differs")
        if metadata["model"] != load_model()["source_sha256"]:
            raise ValueError("External state library model differs")
        if metadata["joint_names"] != list(robot.joint_names):
            raise ValueError("External state library joint order differs")
        if metadata["collection"]["seed"] != cfg.seed or metadata["collection"]["split"] != "eval":
            raise ValueError("External state library is not the requested independent evaluation split")
        env.petbot_dataset = payload["states"]
        validate_dataset(env.petbot_dataset, robot.joint_names, cfg.num_states)
        relative_path = Path(external).name
    else:
        env.petbot_dataset, metadata, relative_path = collect(env, cfg)
    env.petbot_reset_rng = torch.Generator().manual_seed(agent_cfg.seed + (100000 if cfg.split == "eval" else 0))
    env.petbot_eval_mode = cfg.split == "eval"
    env.petbot_eval_cursor = 0
    env.petbot_success_ema = 0.0
    env.petbot_curriculum_pool = reset_pool_indices(env.petbot_dataset["root"])
    if getattr(env.cfg, "sweep2", False):
        if abs(agent_cfg.algorithm.gamma - env.cfg.shaping_gamma) > 1e-9:
            raise ValueError("PPO and potential shaping discounts must match")
        env.petbot_previous_potential = torch.zeros(env.num_envs, device=env.device)
    env.petbot_ready = True
    env.reset()

    env.petbot_provenance = {
        "task": task_name,
        "state_contract": "lab3_xyzw_fresh_body_state_v2",
        "demo_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": False,
        "source_model": load_model(),
        "standing_height_m": env.petbot_standing_height,
        "action_order": list(POSITION_JOINTS + REAR_WHEELS),
        "observation_joint_velocity_order": list(JOINT_ORDER),
        "actor_dim": 27 * getattr(env.cfg, "history_steps", 1),
        "critic_dim": 35 * getattr(env.cfg, "history_steps", 1),
        "physics_dt": env.physics_dt,
        "policy_dt": env.step_dt,
        "success_thresholds": asdict(env.cfg.success),
        "initial_states": relative_path,
        "collection": metadata,
    }
    env.petbot_source_diff = git("diff", "HEAD", "--", ".")
    print(f"[PetBot] default standing height: {env.petbot_standing_height:.5f} m; initial states: {relative_path}")
