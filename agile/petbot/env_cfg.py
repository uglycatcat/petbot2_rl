"""PetBot-only flat-ground recovery task on AGILE / Isaac Lab 3 beta2."""

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .asset_cfg import robot_cfg
from .core import SuccessThresholds
from .model import POSITION_JOINTS, REAR_WHEELS, load_model

SPEC = load_model()


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="min",
        ),
    )
    robot = robot_cfg()
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500))


@configclass
class ActionsCfg:
    # Absolute position targets cover the source limits. Clip is in physical units.
    joint_position = lab_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(POSITION_JOINTS),
        preserve_order=True,
        scale={n: (SPEC["joints"][n]["range"][1] - SPEC["joints"][n]["range"][0]) / 2 for n in POSITION_JOINTS},
        offset={n: sum(SPEC["joints"][n]["range"]) / 2 for n in POSITION_JOINTS},
        use_default_offset=False,
        clip={n: tuple(SPEC["joints"][n]["range"]) for n in POSITION_JOINTS},
    )
    rear_velocity = lab_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(REAR_WHEELS),
        preserve_order=True,
        scale={n: SPEC["actuators"][n]["control_range"][1] for n in REAR_WHEELS},
        use_default_offset=False,
        offset=0.0,
        clip={n: tuple(SPEC["actuators"][n]["control_range"]) for n in REAR_WHEELS},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(func=mdp.actor_observation, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        state = ObsTerm(func=mdp.critic_observation, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class RewardsCfg:
    upright = RewTerm(func=mdp.task_reward, weight=2.0, params={"kind": "upright"})
    configuration = RewTerm(func=mdp.task_reward, weight=3.0, params={"kind": "configuration"})
    height = RewTerm(func=mdp.task_reward, weight=1.0, params={"kind": "height"})
    still = RewTerm(func=mdp.task_reward, weight=0.5, params={"kind": "still"})
    stable = RewTerm(func=mdp.task_reward, weight=5.0, params={"kind": "stable"})
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)
    effort = RewTerm(func=mdp.effort_cost, weight=-0.002)


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_from_dataset, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    invalid = DoneTerm(func=mdp.invalid_state)
    boundary = DoneTerm(func=mdp.out_of_bounds)


@configclass
class EmptyCfg:
    pass


@configclass
class PetBotRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1024, env_spacing=3.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    events = EventsCfg()
    terminations = TerminationsCfg()
    commands = EmptyCfg()
    curriculum = EmptyCfg()
    success = SuccessThresholds()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = SPEC["physics_dt"]
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (0.8, -0.8, 0.6)
        self.viewer.lookat = (0.0, 0.0, 0.1)
