"""PetBot task terms: no humanoid contact rules, mirroring, or lifting action."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .core import HoldTracker, as_torch, observation_vector, success_components, recovery_potential, discounted_progress, gravity_from_xyzw, rotate_inverse_xyzw, initial_classes
from .model import JOINT_ORDER, POSITION_JOINTS, WHEELS


def joint_ids(env, names):
    return env.scene["robot"].find_joints(list(names), preserve_order=True)[0]


def state(env):
    robot = env.scene["robot"]
    data = robot.data
    ids = joint_ids(env, POSITION_JOINTS)
    wheel_ids = joint_ids(env, WHEELS)
    sensor = env.scene["contact_forces"]
    body_ids = sensor.find_bodies(list(WHEELS), preserve_order=True)[0]
    pose = as_torch(data.root_link_pose_w)
    velocity = as_torch(data.root_com_vel_w)
    quaternion = pose[:, 3:7]
    return {
        "gravity": gravity_from_xyzw(quaternion),
        "joint_error": as_torch(data.joint_pos)[:, ids] - as_torch(data.default_joint_pos)[:, ids],
        "lin_vel": rotate_inverse_xyzw(quaternion, velocity[:, :3]),
        "ang_vel": rotate_inverse_xyzw(quaternion, velocity[:, 3:]),
        "pos_vel": as_torch(data.joint_vel)[:, ids],
        "wheel_vel": as_torch(data.joint_vel)[:, wheel_ids],
        "height": pose[:, 2] - as_torch(env.scene.env_origins)[:, 2],
        # Wheels are filtered against robot bodies; positive vertical force is ground support.
        "wheel_forces": as_torch(sensor.data.net_forces_w)[:, body_ids, 2],
    }


def actor_observation(env):
    s = state(env)
    velocity = as_torch(env.scene["robot"].data.joint_vel)[:, joint_ids(env, JOINT_ORDER)]
    return observation_vector(s["gravity"], s["ang_vel"], s["joint_error"], velocity, env.action_manager.action)


def critic_observation(env):
    s = state(env)
    return torch.cat((actor_observation(env), s["lin_vel"], s["height"][:, None], s["wheel_forces"] * 0.1), dim=-1)


def metrics(env):
    if not hasattr(env, "petbot_hold"):
        env.petbot_hold = HoldTracker(env.num_envs, env.step_dt, env.cfg.success.hold_s, env.device)
        env.petbot_metric_step = -1
    if env.petbot_metric_step != env.common_step_counter:
        s = state(env)
        parts = success_components(**s, target_height=env.petbot_standing_height, cfg=env.cfg.success)
        good = torch.stack(list(parts.values())).all(dim=0)
        was_success = env.petbot_hold.ever.clone()
        held = env.petbot_hold.update(good, env.episode_length_buf * env.step_dt)
        env.petbot_new_success = held & ~was_success
        env.petbot_metric_data = (s, parts, good, held)
        env.petbot_metric_step = env.common_step_counter
    return env.petbot_metric_data


def task_reward(env, kind):
    s, _, good, held = metrics(env)
    upright = ((-s["gravity"][:, 2] + 1.0) * 0.5).clamp(0, 1)
    gate = ((upright - 0.75) * 4.0).clamp(0, 1)
    if kind == "upright":
        value = upright
    elif kind == "configuration":
        value = gate * torch.exp(-s["joint_error"].square().mean(dim=-1) / 0.25)
    elif kind == "height":
        value = gate * torch.exp(-(s["height"] - env.petbot_standing_height).square() / 0.03**2)
    elif kind == "still":
        movement = (
            s["lin_vel"].square().sum(dim=-1)
            + 0.1 * s["ang_vel"].square().sum(dim=-1)
            + 0.01 * s["pos_vel"].square().sum(dim=-1)
            + 0.001 * s["wheel_vel"].square().sum(dim=-1)
        )
        value = -gate * movement
    elif kind == "stable":
        value = good.float() + held.float()
    else:
        raise ValueError(kind)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)



def sweep2_reward(env, kind):
    s, _, good, held = metrics(env)
    if kind == "progress":
        potential = recovery_potential(s["gravity"], s["joint_error"])
        value = discounted_progress(env.petbot_previous_potential, potential,
                                    env.cfg.shaping_gamma, env.termination_manager.terminated) / env.step_dt
        env.petbot_previous_potential = potential.detach().clone()
    elif kind == "success":
        value = env.petbot_new_success.float() / env.step_dt
    elif kind == "stable":
        value = good.float() + held.float()
    elif kind == "time":
        value = (~held).float()
    elif kind == "failure":
        value = env.termination_manager.terminated.float() / env.step_dt
    elif kind in ("action_rate", "effort"):
        # Weak during discovery, full regularization after 1000 x 64 control steps.
        scale = 0.1 + 0.9 * min(env.common_step_counter / 64000, 1.0)
        value = scale * (action_rate(env) if kind == "action_rate" else effort_cost(env))
    else:
        raise ValueError(kind)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

def recovery_region(s, height):
    # No reward for lying still: zero at >= 45 degree tilt.
    upright = ((-s["gravity"][:, 2] - .7071) / (.98 - .7071)).clamp(0., 1.)
    configuration = torch.exp(-s["joint_error"].square().mean(-1) / .12)
    elevation = torch.exp(-((s["height"] - height) / .035).square())
    return upright * configuration * elevation


def sweep3_reward(env, kind):
    s, _, _, _ = metrics(env)
    region = recovery_region(s, env.petbot_standing_height)
    speed = (s["lin_vel"].norm(dim=-1) / .10).square()
    speed += (s["ang_vel"].norm(dim=-1) / .3).square()
    speed += (s["pos_vel"].abs().amax(-1) / .3).square()
    speed += (s["wheel_vel"].abs().amax(-1) / 1.).square()
    low_speed = region * torch.exp(-speed)
    if not hasattr(env, "petbot_region_ever"):
        env.petbot_region_ever = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.petbot_low_speed_ever = torch.zeros_like(env.petbot_region_ever)
    env.petbot_region_ever |= region > .8
    env.petbot_low_speed_ever |= low_speed > .5
    return region if kind == "region" else low_speed


def action_rate(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=-1)


def effort_cost(env):
    torque = as_torch(env.scene["robot"].data.applied_torque)[:, joint_ids(env, POSITION_JOINTS)]
    return torch.nan_to_num(torque.square().sum(dim=-1), nan=0.0, posinf=0.0)


def invalid_state(env):
    data = env.scene["robot"].data
    values = torch.cat((as_torch(data.root_state_w), as_torch(data.joint_pos), as_torch(data.joint_vel)), dim=-1)
    local = as_torch(data.root_pos_w) - as_torch(env.scene.env_origins)
    return (
        ~torch.isfinite(values).all(dim=-1)
        | (local[:, 2] < -0.15)
        | (local[:, 2] > 2.0)
        | (as_torch(data.joint_vel).abs().amax(dim=-1) > 150.0)
    )


def out_of_bounds(env):
    xy = as_torch(env.scene["robot"].data.root_pos_w)[:, :2] - as_torch(env.scene.env_origins)[:, :2]
    return xy.norm(dim=-1) > 1.0


def reset_from_dataset(env, env_ids):
    robot = env.scene["robot"]
    if not getattr(env, "petbot_dataset", None):
        if getattr(env, "petbot_ready", False):
            raise RuntimeError("PetBot dataset disappeared after preparation.")
        # Construction/probing only. Training entry must run pre_learn before any rollout.
        root = as_torch(robot.data.default_root_state)[env_ids].clone()
        root[:, :3] += as_torch(env.scene.env_origins)[env_ids]
        position = as_torch(robot.data.default_joint_pos)[env_ids].clone()
        velocity = torch.zeros_like(position)
    else:
        dataset = env.petbot_dataset
        count = len(env_ids)
        if getattr(env, "petbot_eval_mode", False):
            indices = (torch.arange(count) + env.petbot_eval_cursor) % len(dataset["root"])
            env.petbot_eval_cursor += count
        else:
            indices = torch.randint(len(dataset["root"]), (count,), device="cpu", generator=env.petbot_reset_rng)
            if getattr(env.cfg, "easy_reset_curriculum", False) and hasattr(env, "petbot_easy_pool"):
                progress = min(env.petbot_region_ema / .8, env.petbot_low_speed_ema / .5,
                               env.petbot_success_ema / .25, 1.)
                fraction = .5 - .3 * progress
                mask = torch.rand(count, generator=env.petbot_reset_rng) < fraction
                pool = env.petbot_easy_pool
                indices[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), generator=env.petbot_reset_rng)]
                env.petbot_easy_fraction = fraction
            if getattr(env.cfg, "reset_curriculum", False):
                pool = env.petbot_curriculum_pool
                fraction = max(0.2, 0.6 - 0.4 * min(env.petbot_success_ema / 0.35, 1.0))
                mask = torch.rand(count, generator=env.petbot_reset_rng) < fraction
                n = int(mask.sum())
                if len(pool):
                    indices[mask] = pool[torch.randint(len(pool), (n,), generator=env.petbot_reset_rng)]
        if not hasattr(env, "petbot_initial_index"):
            env.petbot_initial_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.petbot_initial_index[env_ids] = indices.to(env.device)
        root = dataset["root"][indices].to(env.device).clone()
        root[:, :3] += as_torch(env.scene.env_origins)[env_ids]
        position = dataset["joint_pos"][indices].to(env.device)
        velocity = dataset["joint_vel"][indices].to(env.device)
    if not hasattr(env, "petbot_initial_class"):
        env.petbot_initial_class = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.petbot_initial_class[env_ids] = initial_classes(root)
    if getattr(env.cfg, "sweep2", False) and hasattr(env, "petbot_previous_potential"):
        gravity = gravity_from_xyzw(root[:, 3:7])
        pos_ids = joint_ids(env, POSITION_JOINTS)
        error = position[:, pos_ids] - as_torch(robot.data.default_joint_pos)[env_ids][:, pos_ids]
        env.petbot_previous_potential[env_ids] = recovery_potential(gravity, error)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
    # No inherited target from the previous episode while waiting for the first policy action.
    robot.set_joint_position_target(position, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(velocity), env_ids=env_ids)


class PetBotRecoveryEnv(ManagerBasedRLEnv):
    """Log real episodic success separately from reward shaping."""

    def step(self, action):
        if not getattr(self, "petbot_ready", False):
            raise RuntimeError("Run the PetBot pre_learn hook before stepping; no standing-only fallback training.")
        result = super().step(action)
        obs, reward, terminated, truncated, extras = result
        # A real failure on the horizon boundary must not receive timeout bootstrap.
        result = (obs, reward, terminated, truncated & ~terminated, extras)
        if getattr(self.cfg, "sweep2", False):
            logs = self.extras.setdefault("log", {})
            logs["Actions/clipped_fraction"] = (action.abs() >= 0.999).float().mean()
            if hasattr(self, "petbot_distribution") and self.petbot_distribution._distribution is not None:
                std = self.petbot_distribution.std.mean(dim=0)
                mean = self.petbot_distribution.mean
                for index, name in enumerate(POSITION_JOINTS + ("left_rear_wheel", "right_rear_wheel")):
                    logs[f"Exploration/std_{name}"] = std[index]
                logs["Actions/mean_outside_range"] = (mean.abs() > 1).float().mean()
                logs["Actions/mean_abs_max"] = mean.abs().max()
                logs["Exploration/min_std"] = self.petbot_distribution.std.min()
            logs["Curriculum/success_ema"] = self.petbot_success_ema
        return result

    def _reset_idx(self, env_ids):
        logs = {}
        if hasattr(self, "petbot_hold"):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = ids[self.episode_length_buf[ids] > 0]
            if len(ids):
                _, parts, _, held = self.petbot_metric_data
                logs["Recovery/success_rate"] = self.petbot_hold.ever[ids].float().mean()
                if hasattr(self, "petbot_success_ema") and not self.petbot_eval_mode:
                    alpha = 1 - (1 - 1/4096) ** len(ids)
                    self.petbot_success_ema = ((1-alpha) * self.petbot_success_ema
                                              + alpha * logs["Recovery/success_rate"].item())
                if hasattr(self, "petbot_region_ever"):
                    for name in ("region", "low_speed"):
                        rate = getattr(self, f"petbot_{name}_ever")[ids].float().mean()
                        logs[f"Recovery/{name}_reached_rate"] = rate
                        if not self.petbot_eval_mode:
                            old = getattr(self, f"petbot_{name}_ema")
                            setattr(self, f"petbot_{name}_ema", (1-alpha)*old + alpha*rate.item())
                    logs["Curriculum/easy_fraction"] = getattr(self, "petbot_easy_fraction", 0.)
                logs["Recovery/stable_at_end"] = held[ids].float().mean()
                times = self.petbot_hold.first_time[ids]
                if torch.isfinite(times).any():
                    logs["Recovery/time_to_success_s"] = times[torch.isfinite(times)].mean()
                if hasattr(self, "petbot_episode_records"):
                    labels = ("upright", "inverted", "side_or_tilted")
                    for idx in ids.tolist():
                        when = self.petbot_hold.first_time[idx].item()
                        self.petbot_episode_records.append(
                            {
                                "initial_class": labels[self.petbot_initial_class[idx].item()],
                                "initial_index": int(self.petbot_initial_index[idx].item()),
                                "env_id": idx,
                                "termination": {name: bool(self.termination_manager.get_term(name)[idx].item())
                                                for name in self.termination_manager.active_terms},
                                "success": bool(self.petbot_hold.ever[idx].item()),
                                "stable_at_end": bool(held[idx].item()),
                                "time_to_success_s": when if when == when else None,
                                "duration_s": self.episode_length_buf[idx].item() * self.step_dt,
                            }
                        )
                for name, condition in parts.items():
                    logs[f"Recovery/final_{name}_rate"] = condition[ids].float().mean()
            self.petbot_hold.reset(env_ids)
            if hasattr(self, "petbot_region_ever"):
                self.petbot_region_ever[env_ids] = False
                self.petbot_low_speed_ever[env_ids] = False
        super()._reset_idx(env_ids)
        self.extras["log"].update(logs)
