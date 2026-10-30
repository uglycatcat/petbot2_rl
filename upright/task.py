"""Isaac Lab adapter for the independent scratch cycle; no legacy task inheritance."""
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as lab_mdp
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from .asset_cfg import robot_cfg
from .config import Config
from .geometry import orientation
from .model import load_model, POSITION_JOINTS, FRONT_WHEELS, REAR_WHEELS
from .runtime import Runtime

SPEC = load_model()


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def ensure(env):
    if not hasattr(env, 'upright_runtime'):
        env.upright_runtime = Runtime(env.num_envs, env.device, env.step_dt, env.cfg.experiment, SPEC)
        env.upright_metric_step = -1
        env.upright_record = False
    return env.upright_runtime


def ids(env):
    if not hasattr(env, 'upright_ids'):
        r, sensor = env.scene['robot'], env.scene['contact_forces']
        env.upright_ids = dict(
            joints=r.find_joints(list(POSITION_JOINTS), preserve_order=True)[0],
            wheels=r.find_joints(list(REAR_WHEELS), preserve_order=True)[0],
            front=r.find_bodies(list(FRONT_WHEELS), preserve_order=True)[0],
            rear=r.find_bodies(list(REAR_WHEELS), preserve_order=True)[0],
            sf=sensor.find_bodies(list(FRONT_WHEELS), preserve_order=True)[0],
            sr=sensor.find_bodies(list(REAR_WHEELS), preserve_order=True)[0],
            other=[i for i, name in enumerate(sensor.body_names) if name not in FRONT_WHEELS+REAR_WHEELS])
    return env.upright_ids


def state(env):
    d, index = env.scene['robot'].data, ids(env)
    root = tensor(d.root_state_w)
    g, yaw, gyro, motion = orientation(root[:, 3:7], root[:, 7:10], root[:, 10:13])
    q = tensor(d.joint_pos)[:, index['joints']]
    gap, skew = ensure(env).geometry.estimate(g, q[:, 3:])
    # Root velocity stands in for a motion estimator. It is explicitly recorded as
    # simulator-derived; no real-hardware estimator accuracy is claimed here.
    return dict(g=g, yaw=yaw, gyro=gyro, motion=motion, q=q,
                qd=tensor(d.joint_vel)[:, index['joints']], wheel=tensor(d.joint_vel)[:, index['wheels']],
                gap=gap, rear_skew=skew)


def truth(env):
    r, index = env.scene['robot'], ids(env)
    heights = tensor(r.data.body_pos_w)[:, :, 2]-tensor(env.scene.env_origins)[:, 2, None]
    history = tensor(env.scene['contact_forces'].data.net_forces_w_history)
    if history.shape[1] != env.cfg.decimation:
        raise RuntimeError('Contact history must cover every physics substep')
    body_contact = (history[:, :, index['other']].norm(dim=-1) > .1).any(-1).any(-1)
    rear_contact = (history[:, :, index['sr'], 2] > .05).all(-1)
    front_contact = (history[:, :, index['sf'], 2] > .05).all(-1)
    return dict(front_z=heights[:, index['front']], rear_z=heights[:, index['rear']],
                root_height=tensor(r.data.root_state_w)[:, 2]-tensor(env.scene.env_origins)[:, 2],
                body_contact=body_contact, rear_fraction=rear_contact.float().mean(-1),
                four_fraction=(rear_contact & front_contact).float().mean(-1),
                front_force_peak=history[:, :, index['sf']].norm(dim=-1).amax(dim=(1, 2)),
                position=tensor(r.data.root_state_w)[:, :3]-tensor(env.scene.env_origins),
                quaternion=tensor(r.data.root_state_w)[:, 3:7],
                effort=tensor(r.data.applied_torque)[:, index['joints']+index['wheels']])


class CycleAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.position_ids = self.robot.find_joints(list(POSITION_JOINTS), preserve_order=True)[0]
        self.wheel_ids = self.robot.find_joints(list(REAR_WHEELS), preserve_order=True)[0]
        self._raw = torch.zeros(env.num_envs, 3, device=env.device)
        self.position = tensor(self.robot.data.default_joint_pos)[:, self.position_ids].clone()
        self.velocity = torch.zeros(env.num_envs, 2, device=env.device)

    @property
    def action_dim(self): return 3
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return torch.cat((self.position, self.velocity), -1)

    def process_actions(self, actions):
        p = ensure(self._env); s = state(self._env)
        self._raw.copy_(actions)
        target = p.control.targets(actions, s)
        self.position[:, 3:] = target[:, :1]
        self.velocity.copy_(target[:, 1:])

    def apply_actions(self):
        self.robot.set_joint_position_target(self.position, joint_ids=self.position_ids)
        self.robot.set_joint_velocity_target(self.velocity, joint_ids=self.wheel_ids)

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self._raw[selected] = 0
        self.position[selected] = tensor(self.robot.data.default_joint_pos)[selected][:, self.position_ids]
        self.velocity[selected] = 0


@configclass
class CycleActionCfg(ActionTermCfg):
    class_type: type = CycleAction
    asset_name: str = 'robot'


def metrics(env):
    p = ensure(env)
    if env.upright_metric_step != env.common_step_counter:
        s, t = state(env), truth(env)
        result = p.advance(s, t)
        # Schedule the next command/Space event before the next observation is
        # produced, so the actor sees the mode for which its action will execute.
        new_calls = p.before_action(s)
        if len(new_calls):
            env.observation_manager.reset(new_calls)
        env.upright_metric_step = env.common_step_counter
        env.upright_metrics = result
        env.extras.setdefault('log', {}).update({
            'Cycle/true_support': result['support_truth'].float().mean(),
            'Cycle/complete': result['complete'].float().mean(),
            'Cycle/stage_goal': result['goal'].float().mean(),
            'Cycle/perception_mismatch': result['perception_mismatch'].float().mean(),
            'Cycle/course_level': float(p.course.level),
            'Cycle/early_front_return': result['early_front_return'].float().mean(),
            'Cycle/failure_penalty': result['failure_penalty'].mean(),
            'Physics/rear_contact_fraction': t['rear_fraction'].mean(),
            'Physics/wheel_actual_peak': s['wheel'].abs().max(),
            'Physics/wheel_target_peak': p.control.executed[:, 1:].abs().max(),
        })
        env.extras['log'].update(getattr(env, 'upright_episode_log', {}))
        if env.upright_record:
            env.upright_snapshot = {k: v.clone() for k, v in dict(
                **s, **t, **result, phase=p.control.phase, requested_command=p.control.requested_command,
                effective_command=torch.where(((p.control.phase == 1) | (p.control.phase == 3))[:, None],
                                               p.control.transition_motion, p.control.command),
                requested_action=p.control.requested_action, executed_targets=p.control.executed,
                stage_time=p.control.age, longest_hold=p.longest_hold, curriculum_level=p.level).items()}
    return env.upright_metrics


def observation(env):
    return ensure(env).observation(state(env))


def reward(env):
    return metrics(env)['reward']/env.step_dt  # manager applies dt once


def done(env):
    return metrics(env)['done']


def reset(env, env_ids):
    p, r = ensure(env), env.scene['robot']
    if len(env_ids) == 0:
        return
    env.upright_episode_log = {
        'Cycle/episode_balance': p.balance_seen[env_ids].float().mean(),
        'Cycle/episode_complete': p.true_cycle[env_ids].float().mean(),
        'Cycle/episode_stage_goal': p.episode_goal[env_ids].float().mean(),
        'Cycle/episode_failure': p.failure[env_ids].float().mean(),
    }
    root = tensor(r.data.default_root_state)[env_ids].clone()
    root[:, :3] += tensor(env.scene.env_origins)[env_ids]; root[:, 7:] = 0
    q = tensor(r.data.default_joint_pos)[env_ids].clone()
    r.write_root_state_to_sim(root, env_ids=env_ids)
    r.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    # Joint variation is reached using ordinary control during warmup, never by
    # teleporting the hip/body state at the moment of Space takeover.
    p.reset(env_ids)
    # Do not invalidate the current post-step metric cache: termination/reward
    # may be read again after the automatic reset. Next physics step advances it.


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path='/World/ground', terrain_type='plane', collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5,
                                                       restitution=0., friction_combine_mode='min'))
    robot = robot_cfg()
    contact_forces = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*', history_length=30, update_period=.002)

@configclass
class ActionsCfg:
    cycle = CycleActionCfg()

@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=observation)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True
    policy = Policy()

@configclass
class RewardsCfg:
    cycle = RewardTermCfg(func=reward, weight=1.)

@configclass
class EventsCfg:
    reset_robot = EventTermCfg(func=reset, mode='reset')

@configclass
class DonesCfg:
    completed_or_failed = TerminationTermCfg(func=done)
    timeout = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)

@configclass
class EmptyCfg:
    pass

@configclass
class UprightEnvCfg(ManagerBasedRLEnvCfg):
    experiment = Config()
    scene = SceneCfg(num_envs=2048, env_spacing=6., lazy_sensor_update=False)
    actions = ActionsCfg(); observations = ObservationsCfg(); rewards = RewardsCfg()
    events = EventsCfg(); terminations = DonesCfg(); commands = EmptyCfg(); curriculum = EmptyCfg()
    def __post_init__(self):
        super().__post_init__()
        self.apply_experiment(self.experiment)

    def apply_experiment(self, experiment):
        self.experiment = experiment
        self.seed = experiment.seed
        self.scene.num_envs = experiment.num_envs
        self.decimation = experiment.decimation
        self.episode_length_s = (experiment.warmup_timeout_s+experiment.rise_timeout_s+
                                max(experiment.balance_eval_s, experiment.balance_train_s)+
                                experiment.landing_timeout_s+experiment.handoff_timeout_s+1)
        self.sim.dt = SPEC['physics_dt']; self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_max_rigid_patch_count = 10*2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = self.decimation
        self.scene.lazy_sensor_update = False
        self.observations.policy.history_length = experiment.history
