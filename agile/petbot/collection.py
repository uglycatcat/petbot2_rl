"""AGILE-style pre-collected reset library, adapted for passive wheels and implicit drives."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from .core import as_torch, sample_joints, uniform_quaternions
from .model import POSITION_JOINTS, PROJECT_ROOT, load_model, prepared_asset


@dataclass
class CollectionCfg:
    num_states: int = 16384
    seed: int = 20261013
    split: str = "train"
    spawn_height: float = 0.6
    settle_s: float = 3.0
    hold_s: float = 0.5
    max_batches: int = 128
    max_linear_speed: float = 0.05
    max_angular_speed: float = 0.2
    max_joint_speed: float = 0.5


def cache_identity(env, cfg):
    usd, _ = prepared_asset()
    payload = {
        "schema": 2,
        "quaternion_order": "xyzw",
        "root_velocity_frame": "world_com",
        "model": load_model()["source_sha256"],
        "collection": asdict(cfg),
        "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest(),
        "physics_dt": env.physics_dt,
        "decimation": env.cfg.decimation,
        "num_envs": env.num_envs,
        "joint_names": list(env.scene["robot"].joint_names),
        "implementation": {
            name: hashlib.sha256((PROJECT_ROOT / "agile/petbot" / name).read_bytes()).hexdigest()
            for name in ("collection.py", "core.py", "asset_cfg.py", "env_cfg.py")
        },
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return key, payload


@contextmanager
def disabled_drives(robot):
    stiffness = as_torch(robot.data.joint_stiffness).clone()
    damping = as_torch(robot.data.joint_damping).clone()
    try:
        # Zeroing efforts alone would leave position/velocity drives active.
        robot.write_joint_stiffness_to_sim(torch.zeros_like(stiffness))
        robot.write_joint_damping_to_sim(torch.zeros_like(damping))
        robot.set_joint_effort_target(torch.zeros_like(stiffness))
        yield
    finally:
        robot.write_joint_stiffness_to_sim(stiffness)
        robot.write_joint_damping_to_sim(damping)


def physics_step(env):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def write_state(env, root, position, velocity):
    robot = env.scene["robot"]
    env.scene.reset()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(position, velocity)
    robot.set_joint_position_target(position)
    robot.set_joint_velocity_target(torch.zeros_like(velocity))
    robot.set_joint_effort_target(torch.zeros_like(velocity))


def probe_standing(env):
    """Measure this asset's settled default height; never borrow another robot's target."""
    from .mdp import state

    robot = env.scene["robot"]
    root = as_torch(robot.data.default_root_state).clone()
    root[:, :3] += as_torch(env.scene.env_origins)
    position = as_torch(robot.data.default_joint_pos).clone()
    write_state(env, root, position, torch.zeros_like(position))
    consecutive = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for _ in range(math.ceil(3.0 / env.physics_dt)):
        physics_step(env)
        s = state(env)
        good = (
            (-s["gravity"][:, 2] > 0.98)
            & (s["joint_error"].abs().amax(-1) < 0.15)
            & (s["lin_vel"].norm(dim=-1) < 0.05)
            & (s["ang_vel"].norm(dim=-1) < 0.15)
            & (s["pos_vel"].abs().amax(-1) < 0.15)
            & (s["wheel_vel"].abs().amax(-1) < 0.5)
            & (s["wheel_forces"] > env.cfg.success.contact_force).all(-1)
        )
        consecutive = torch.where(good, consecutive + 1, 0)
    valid = consecutive >= math.ceil(0.5 / env.physics_dt)
    if valid.float().mean() < 0.9:
        raise RuntimeError(
            f"Default stance failed: {valid.float().mean().item():.1%} stable. "
            "Inspect asset import, contact geometry and servo gains before training."
        )
    return float(s["height"][valid].median().item())


def validate_dataset(dataset, names, minimum_count):
    required = {"root": 13, "joint_pos": len(names), "joint_vel": len(names)}
    counts = set()
    for key, width in required.items():
        data = dataset[key]
        if data.ndim != 2 or data.shape[1] != width or not torch.isfinite(data).all():
            raise ValueError(f"Invalid fallen dataset field: {key}")
        counts.add(len(data))
    if len(counts) != 1 or next(iter(counts)) < minimum_count:
        raise ValueError("Incomplete fallen dataset")
    if not torch.allclose(dataset["root"][:, 3:7].norm(dim=-1), torch.ones(len(dataset["root"])), atol=1e-4):
        raise ValueError("Invalid fallen quaternions")


def collect(env, cfg):
    if cfg.split not in {"train", "eval"} or cfg.num_states < 1 or cfg.settle_s < cfg.hold_s:
        raise ValueError("Invalid initial-state collection configuration")
    if getattr(env, "_physics_handles_decimation", False):
        raise RuntimeError("PetBot collection requires PhysX substeps, not backend-managed decimation.")
    key, metadata = cache_identity(env, cfg)
    directory = PROJECT_ROOT / "artifacts/fallen_states"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cfg.split}_{key}.pt"
    robot = env.scene["robot"]
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"] != metadata:
            raise ValueError("Fallen cache identity mismatch")
        validate_dataset(payload["states"], robot.joint_names, cfg.num_states)
        return payload["states"], metadata, str(path.relative_to(PROJECT_ROOT))

    spec = load_model()
    rng = torch.Generator(device=env.device).manual_seed(cfg.seed)
    samples = {k: [] for k in ("root", "joint_pos", "joint_vel")}
    count = 0
    required = math.ceil(cfg.hold_s / env.physics_dt)
    with disabled_drives(robot):
        for batch in range(cfg.max_batches):
            root = as_torch(robot.data.default_root_state).clone()
            root[:, :3] = as_torch(env.scene.env_origins)
            root[:, 2] += cfg.spawn_height
            root[:, 3:7] = uniform_quaternions(env.num_envs, env.device, rng)
            root[:, 7:] = 0
            position = sample_joints(env.num_envs, robot.joint_names, spec, env.device, rng)
            write_state(env, root, position, torch.zeros_like(position))
            consecutive = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            for _ in range(math.ceil(cfg.settle_s / env.physics_dt)):
                physics_step(env)
                r = as_torch(robot.data.root_state_w)
                q = as_torch(robot.data.joint_pos)
                v = as_torch(robot.data.joint_vel)
                local_height = r[:, 2] - as_torch(env.scene.env_origins)[:, 2]
                finite = torch.isfinite(torch.cat((r, q, v), -1)).all(-1)
                # Require actual settled states; do not turn moving samples into stationary ones by zeroing velocity.
                stable = (
                    finite
                    & (local_height > 0.005)
                    & (local_height < cfg.spawn_height)
                    & (r[:, 7:10].norm(dim=-1) < cfg.max_linear_speed)
                    & (r[:, 10:13].norm(dim=-1) < cfg.max_angular_speed)
                    & (v.abs().amax(-1) < cfg.max_joint_speed)
                )
                body_z = as_torch(robot.data.body_pos_w)[:, :, 2] - as_torch(env.scene.env_origins)[:, None, 2]
                stable &= body_z.amin(-1) > -0.03
                forces = as_torch(env.scene["contact_forces"].data.net_forces_w)
                stable &= forces[:, :, 2].sum(-1) > 0.5
                for name in POSITION_JOINTS:
                    j = robot.joint_names.index(name)
                    low, high = spec["joints"][name]["range"]
                    stable &= (q[:, j] >= low - 0.02) & (q[:, j] <= high + 0.02)
                consecutive = torch.where(stable, consecutive + 1, 0)
            ids = torch.where(consecutive >= required)[0][: cfg.num_states - count]
            if len(ids):
                states = r[ids].clone()
                states[:, :3] -= as_torch(env.scene.env_origins)[ids]
                # Plane is translationally invariant. Center each initial state inside its own cell.
                states[:, :2] = 0
                samples["root"].append(states.cpu())
                samples["joint_pos"].append(q[ids].clone().cpu())
                samples["joint_vel"].append(v[ids].clone().cpu())
                count += len(ids)
            print(f"[PetBot] {cfg.split} initial states: {count}/{cfg.num_states}, batch {batch + 1}", flush=True)
            if count >= cfg.num_states:
                break
    if count < cfg.num_states:
        raise RuntimeError(f"Only {count}/{cfg.num_states} settled states; inspect contacts/collection settings.")
    dataset = {k: torch.cat(v) for k, v in samples.items()}
    validate_dataset(dataset, robot.joint_names, cfg.num_states)
    temporary = path.with_suffix(".tmp")
    torch.save({"metadata": metadata, "states": dataset}, temporary)
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dataset, metadata, str(path.relative_to(PROJECT_ROOT))
