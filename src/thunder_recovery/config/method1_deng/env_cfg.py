# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder fall-recovery environment — standalone, no robot_lab dependency.

Based on 'Learning to Recover' (arXiv:2506.05516).

Design notes:
- Scene: flat plane (recovery does not use rough terrain).
- Actor obs: 10-frame history on base_ang_vel, projected_gravity,
  joint_pos_rel, joint_vel_rel, last_action; base_lin_vel from IMU (noisy).
- Critic obs: same actor stack (clean) + privileged priv_* (instant).
- No velocity commands, no height scan — the recovery policy is
  command-free by design.
- Events: reset_with_freefall + zero_action_freefall (recovery core), plus
  light DR on base mass (+-10 %), external force/torque on reset, startup
  friction randomisation.
- Rewards: 13 terms (paper Table I + support_state + wheel-leg coord).
- Terminations: timeout only (recovery never ends early).
- No curriculum.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_core
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from thunder_recovery import mdp as recovery_mdp
from thunder_recovery.config.asset import THUNDER_NOHEAD_CFG


# -- Joint / body naming ------------------------------------------------
LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
BASE_LINK_NAME = "base_link"
FOOT_LINK_REGEX = ".*_foot"          # bodies
COLLISION_BODY_REGEXES = [BASE_LINK_NAME, ".*thigh.*", ".*calf.*"]


# -- Scene --------------------------------------------------------------
@configclass
class RecoverySceneCfg(InteractiveSceneCfg):
    """Flat plane + Thunder robot + contact sensor. Minimal by design."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot: filled in via replace() to get {ENV_REGEX_NS} per-env prim path.
    robot: ArticulationCfg = THUNDER_NOHEAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    # Height scanner: privileged, critic-only. Centered 20 m above the base,
    # ray-cast onto /World/ground. On the plane terrain used here the scan is
    # flat, but keeping the sensor wired in means no code changes are needed
    # when extending to rough terrain or benchmarking sim-to-real — the critic
    # already consumes the signal.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + BASE_LINK_NAME,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# -- Actions ------------------------------------------------------------
@configclass
class RecoveryActionsCfg:
    """Position control for legs, velocity control for wheels.

    Scales mirror the ThunderHist baseline:
      hip joints:   0.125 (narrow range, hips do not need big swings)
      thigh/calf:   0.25
      wheels:       0.8   (paper Section C: reduced wheel scale for recovery)
    """

    joint_pos = mdp_core.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        # 论文/作者实现：作者部署栈（Recovery_go2w runtest.py）用统一的
        # actions*0.5 + default_dof_pos；上游原实现用 hip 0.125 / thigh·calf 0.25，
        # 探索幅度只有一半到四分之一，是复现失败的首要嫌疑。
        scale=0.5,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    joint_vel = mdp_core.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=0.5,   # 作者栈同样用 0.5（论文只说 s_v < 1.0）
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


# -- Observations -------------------------------------------------------
@configclass
class RecoveryObservationsCfg:
    """Asymmetric actor-critic observations — paper author spec.

    Reference: Recovery_go2w/simulate_python/test/runtest.py (the 78-dim
    Actor class that loads the paper's `model_7999.pt` checkpoint).

    Actor (78 dim, 9 terms, history_length=1):
      pre_actions(16) + projected_gravity(3) + base_ang_vel(3) +
      joint_pos_legs(12) + joint_vel_legs(12) + wheel_vel(4) +
      previous_joint_pos_legs(12) + previous_joint_vel_legs(12) +
      previous_wheel_vel(4).
      No base_lin_vel — paper does not trust IMU integration drift.

    Critic (~262 dim, 15 terms): actor 9 terms (denoised) + privileged:
      priv_base_lin_vel(3) + priv_base_ang_vel(3) + priv_base_height(1) +
      priv_foot_contact(4) + priv_body_contact_force(~9) + height_scan(~187).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Order matches paper runtest.py Actor class signature.
        pre_actions = ObsTerm(
            func=mdp_core.last_action,
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp_core.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
        )
        base_ang_vel = ObsTerm(
            func=mdp_core.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
        )
        joint_pos_legs = ObsTerm(
            func=recovery_mdp.joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel_legs = ObsTerm(
            func=recovery_mdp.joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
        )
        wheel_vel = ObsTerm(
            func=recovery_mdp.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
        )
        previous_joint_pos_legs = ObsTerm(
            func=recovery_mdp.previous_joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        previous_joint_vel_legs = ObsTerm(
            func=recovery_mdp.previous_joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        previous_wheel_vel = ObsTerm(
            func=recovery_mdp.previous_wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # 1-9: clean mirror of actor obs (no noise, history_length=1).
        pre_actions = ObsTerm(func=mdp_core.last_action, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp_core.projected_gravity, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp_core.base_ang_vel, clip=(-100.0, 100.0))
        joint_pos_legs = ObsTerm(
            func=recovery_mdp.joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        joint_vel_legs = ObsTerm(
            func=recovery_mdp.joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        wheel_vel = ObsTerm(
            func=recovery_mdp.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        previous_joint_pos_legs = ObsTerm(
            func=recovery_mdp.previous_joint_pos_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        previous_joint_vel_legs = ObsTerm(
            func=recovery_mdp.previous_joint_vel_legs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        previous_wheel_vel = ObsTerm(
            func=recovery_mdp.previous_wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )

        # 10-14: Privileged (instant, sim-ground-truth).
        priv_base_lin_vel = ObsTerm(
            func=recovery_mdp.priv_base_lin_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        priv_base_ang_vel = ObsTerm(
            func=recovery_mdp.priv_base_ang_vel_clean,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )
        priv_base_height = ObsTerm(
            func=recovery_mdp.priv_base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-5.0, 5.0),
        )
        priv_foot_contact = ObsTerm(
            func=recovery_mdp.priv_foot_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_REGEX),
                "threshold": 1.0,
            },
            clip=(0.0, 1.0),
        )
        priv_body_contact_force = ObsTerm(
            func=recovery_mdp.priv_body_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_BODY_REGEXES)},
            clip=(0.0, 500.0),
            scale=0.01,
        )

        # 15: Flat-terrain scan today, but wiring stays intact for rough terrain.
        height_scan = ObsTerm(
            func=mdp_core.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# -- Events -------------------------------------------------------------
@configclass
class RecoveryEventsCfg:
    """Reset + DR + free-fall torque override.

    Order of reset-mode events is important: `reset_recovery` must run
    before `freefall_zero_action_on_reset` so the step counter is
    initialised before the gain-zeroing logic reads it. Python dataclass
    field order is preserved by isaaclab's reward/event manager.
    """

    # 1. Startup: friction randomisation
    randomize_friction = EventTerm(
        func=mdp_core.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),   # 论文 §IV-A：静摩擦 U[0.7, 1.3]
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # 2. Reset: recovery random-fallen init (paper Section III-A)
    reset_recovery = EventTerm(
        func=recovery_mdp.reset_with_freefall,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "drop_height": 1.1},
    )

    # 3. Reset: zero actuator gains at t=0 (prevent first-step PD impulse)
    freefall_zero_action_on_reset = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 4. Reset: 论文 §IV-A 的域随机化 —— 基座质量 ±5 kg、连杆质量 ±10%、
    #    执行器刚度 ±15 / 阻尼 ±2（刚度基准 100 → 0.85~1.15 倍；阻尼基准 5 → 0.6~1.4 倍）
    randomize_base_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    randomize_link_mass = EventTerm(
        func=mdp_core.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp_core.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.6, 1.4),
        },
    )

    # 6. Interval (every control step): keep gains zero during free-fall
    freefall_zero_action = EventTerm(
        func=recovery_mdp.zero_action_freefall,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 7. (已移除) 周期性推力：论文 §IV-A 的 DR 清单里没有它


# -- Rewards (paper Table I + support_state + wheel-leg coord) ----------
@configclass
class RecoveryRewardsCfg:
    # Step counter: advances ED clock (side-effect term).
    recovery_step_counter = RewTerm(
        func=recovery_mdp.recovery_step_counter,
        weight=1e-6,
    )

    # Logging only: success indicator over last 1 s of episode.
    recovery_success_rate = RewTerm(
        func=recovery_mdp.recovery_success_rate,
        weight=1e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Task rewards (xED, paper Table I)
    recovery_stand_joint_pos = RewTerm(
        func=recovery_mdp.recovery_stand_joint_pos,
        weight=42.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
    )
    recovery_base_height = RewTerm(
        func=recovery_mdp.recovery_base_height,
        weight=120.0,
        params={"target_height": 0.426, "sigma": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_base_orientation = RewTerm(
        func=recovery_mdp.recovery_base_orientation,
        # Table I 的 scale 是 50、定义式是 (g_b − e_z)^2（惩罚）→ 权重取 −50
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Behavior rewards（论文 Table I 原值；上游曾整体下调 5~7 倍）
    recovery_body_collision = RewTerm(
        func=recovery_mdp.recovery_body_collision,
        weight=-5e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=COLLISION_BODY_REGEXES),
            "force_clip": 50.0,
        },
    )
    recovery_action_rate_legs = RewTerm(
        func=recovery_mdp.recovery_action_rate_legs,
        weight=-1e-2,
    )

    # Support state — 4 feet contact AND base held above min_base_height.
    # The height gate is critical for wheeled-legged robots: without it the
    # policy converges to belly-flat with wheels grounded to collect this
    # reward without standing up. 0.30 m matches the success criteria,
    # so earning support_state now requires visible progress toward success.
    recovery_support_state = RewTerm(
        func=recovery_mdp.recovery_support_state,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_REGEX),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 1.0,
            "min_base_height": 0.30,
        },
    )

    # Wheel-leg coordination (paper core contribution, early-phase only)
    recovery_wheel_leg_coord = RewTerm(
        func=recovery_mdp.recovery_wheel_leg_coord,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_wheel_speed": 40.0},
    )

    # （已移除）recovery_joint_deviation：该项不在论文 Table I / Eq.4 中，
    # 是上游自加的常驻关节偏差惩罚，会与论文的 (×DS) stand_joint_pos 叠加。

    # 常量惩罚（论文 Table I 原值；上游曾以"防止低运动量局部最优"为下调至 1/5~1/7）
    recovery_joint_velocity = RewTerm(
        func=recovery_mdp.recovery_joint_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_torques = RewTerm(
        func=recovery_mdp.recovery_torques,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_joint_acceleration = RewTerm(
        func=recovery_mdp.recovery_joint_acceleration,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    recovery_wheel_velocity = RewTerm(
        func=recovery_mdp.recovery_wheel_velocity,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# -- Terminations -------------------------------------------------------
@configclass
class RecoveryTerminationsCfg:
    """Timeout only — recovery never ends early (paper Section III)."""
    time_out = DoneTerm(func=mdp_core.time_out, time_out=True)


# -- Main env cfg -------------------------------------------------------
@configclass
class ThunderRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone recovery env — 4 s episodes (paper §IV-A), 50 Hz control, flat plane."""

    # Scene
    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: RecoveryObservationsCfg = RecoveryObservationsCfg()
    actions: RecoveryActionsCfg = RecoveryActionsCfg()
    events: RecoveryEventsCfg = RecoveryEventsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()

    # No commands (recovery is command-free by design).
    # No curriculum (paper does not use one).

    def __post_init__(self):
        # -- Sim --
        # 论文 §IV-A：episode 4 s、控制 100 Hz。本环境物理步 0.005 s × decimation 4
        # = 0.02 s（50 Hz），与作者部署栈的策略周期（0.002 s × 10 步）一致，
        # 因此保持 50 Hz、把 episode 长度改为论文的 4 s（200 控制步）。
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # -- Viewer --
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
