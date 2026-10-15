"""Symmetric rear-wheel balance, with default standing resets and no demonstration data."""
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp as lab_mdp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from .asset_cfg import robot_cfg
from .model import load_model, POSITION_JOINTS, REAR_WHEELS, FRONT_WHEELS

SPEC = load_model()
def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch

class SymmetricAction(ActionTerm):
    """Two bounded channels expanded to symmetric physical targets."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self.pos_ids = self.robot.find_joints(list(POSITION_JOINTS), preserve_order=True)[0]
        self.wheel_ids = self.robot.find_joints(list(REAR_WHEELS), preserve_order=True)[0]
        self._raw = torch.zeros(env.num_envs, 2, device=env.device)
        self.target = tensor(self.robot.data.default_joint_pos)[:, self.pos_ids].clone()
        self.speed = torch.zeros(env.num_envs, 2, device=env.device)
        self.lo, self.hi = SPEC['joints']['left_frame']['range']
        assert SPEC['joints']['right_frame']['range'] == [self.lo, self.hi]
    @property
    def action_dim(self): return 2
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return torch.cat((self.target, self.speed), -1)
    def process_actions(self, actions):
        if not torch.isfinite(actions).all(): raise RuntimeError('Nonfinite policy action')
        if actions.abs().max() > 1.00001: raise RuntimeError('Policy action outside bounded contract')
        self._raw[:] = actions
        frame = self.lo + (actions[:, 0]+1)*.5*(self.hi-self.lo)
        self.target[:, 3:] = frame[:, None]
        self.speed[:] = (actions[:, 1]*SPEC['actuators']['left_rear_wheel']['control_range'][1])[:, None]
    def apply_actions(self):
        self.robot.set_joint_position_target(self.target, joint_ids=self.pos_ids)
        self.robot.set_joint_velocity_target(self.speed, joint_ids=self.wheel_ids)
    def reset(self, env_ids=None):
        self._raw[env_ids] = 0
        self.target[env_ids] = tensor(self.robot.data.default_joint_pos)[env_ids][:, self.pos_ids] if env_ids is not None else tensor(self.robot.data.default_joint_pos)[:,self.pos_ids]
        self.speed[env_ids] = 0

@configclass
class SymmetricActionCfg(ActionTermCfg):
    class_type: type = SymmetricAction
    asset_name: str = 'robot'

def state(env):
    r=env.scene['robot']; d=r.data
    if not hasattr(env,'upright_ids'):
        front=r.find_bodies(list(FRONT_WHEELS),preserve_order=True)[0]
        rear=r.find_bodies(list(REAR_WHEELS),preserve_order=True)[0]
        sensor=env.scene['contact_forces']
        sf=sensor.find_bodies(list(FRONT_WHEELS),preserve_order=True)[0]
        sr=sensor.find_bodies(list(REAR_WHEELS),preserve_order=True)[0]
        other=[i for i,n in enumerate(sensor.body_names) if n not in REAR_WHEELS]
        joints=r.find_joints(list(POSITION_JOINTS),preserve_order=True)[0]
        wheels=r.find_joints(list(REAR_WHEELS),preserve_order=True)[0]
        env.upright_ids=(front,rear,sf,sr,other,joints,wheels)
    front,rear,sf,sr,other,joints,wheels=env.upright_ids
    root=tensor(d.root_state_w); x,y,z,w=root[:,3:7].unbind(-1)
    gravity=torch.stack((2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1),-1)
    pos=root[:,:3]-tensor(env.scene.env_origins)
    forces=tensor(env.scene['contact_forces'].data.net_forces_w)
    heights=tensor(d.body_pos_w)[:,:,2]-tensor(env.scene.env_origins)[:,2,None]
    return dict(g=gravity,pos=pos,vel=root[:,7:13],q=tensor(d.joint_pos)[:,joints],qd=tensor(d.joint_vel)[:,joints],
                wheel=tensor(d.joint_vel)[:,wheels],front_z=heights[:,front].amin(-1),rear_z=heights[:,rear].amax(-1),
                rear=(forces[:,sr,2]>.05).all(-1), other=(forces[:,other].norm(dim=-1)>.1).any(-1),
                front=(forces[:,sf].norm(dim=-1)>.1).any(-1))

def observations(env):
    s=state(env)
    return torch.cat((s['g'],s['vel'],s['q'],s['qd']*.1,s['wheel']*.05,s['pos'][:,:2],env.action_manager.action),-1)

def metrics(env):
    if not hasattr(env,'upright_hold'):
        env.upright_hold=torch.zeros(env.num_envs,device=env.device)
        env.upright_ever=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        env.upright_step=-1
    if env.upright_step != env.common_step_counter:
        s=state(env)
        # Height plus contact rejects contact dropout and four-wheel standing.
        support=s['rear'] & ~s['other'] & (s['front_z']>.10)
        calm=(s['vel'][:,:3].norm(dim=-1)<.15)&(s['vel'][:,3:].norm(dim=-1)<.4)&(s['qd'].abs().amax(-1)<.5)
        good=support & calm & (s['g'][:,2]<-.25)
        env.upright_hold=torch.where(good,env.upright_hold+env.step_dt,0.)
        held=env.upright_hold>=1.
        fresh=held & ~env.upright_ever
        env.upright_ever |= held
        env.upright_metrics=(s,support,good,held,fresh)
        env.upright_step=env.common_step_counter
        env.extras.setdefault('log',{}).update({'Upright/support':support.float().mean(),'Upright/held':held.float().mean(), 'Upright/front_height':s['front_z'].mean()})
    return env.upright_metrics

def reward(env,kind):
    s,support,good,held,fresh=metrics(env)
    lift=((s['front_z']-.04)/.10).clamp(0,1)*s['rear'].float()
    if kind=='lift': return lift
    if kind=='support': return support.float()
    if kind=='balance': return support.float()*torch.exp(-s['vel'][:,:3].square().sum(-1)/.04-s['vel'][:,3:].square().sum(-1)/.25)
    if kind=='success': return fresh.float()/env.step_dt
    if kind=='position': return s['pos'][:,:2].square().sum(-1).clamp(max=4)
    if kind=='rate': return (env.action_manager.action-env.action_manager.prev_action).square().sum(-1)
    if kind=='fall': return env.termination_manager.terminated.float()/env.step_dt
    raise ValueError(kind)

def failed(env):
    s=state(env)
    return ~torch.isfinite(torch.cat((s['pos'],s['vel'],s['q']),-1)).all(-1) | (s['pos'][:,:2].norm(dim=-1)>1.5) | (s['pos'][:,2]<.025) | (s['g'][:,2]>.2)

def reset(env,env_ids):
    r=env.scene['robot']
    root=tensor(r.data.default_root_state)[env_ids].clone()
    root[:,:3]+=tensor(env.scene.env_origins)[env_ids]
    root[:,7:]=0
    r.write_root_state_to_sim(root,env_ids=env_ids)
    q=tensor(r.data.default_joint_pos)[env_ids].clone()
    r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
    if hasattr(env,'upright_hold'):
        env.extras.setdefault('log',{})['Upright/episode_success']=env.upright_ever[env_ids].float().mean()
        env.upright_hold[env_ids]=0;env.upright_ever[env_ids]=False;env.upright_step=-1

@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain=TerrainImporterCfg(prim_path='/World/ground',terrain_type='plane',collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5,dynamic_friction=1.5,restitution=0.,friction_combine_mode='min'))
    robot=robot_cfg()
    contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=1)
@configclass
class ActionsCfg:
    symmetric=SymmetricActionCfg()
@configclass
class ObsCfg:
    @configclass
    class Policy(ObsGroup):
        state=ObsTerm(func=observations)
        def __post_init__(self):
            self.enable_corruption=False;self.concatenate_terms=True
            self.history_length=5;self.flatten_history_dim=True
    policy=Policy()
@configclass
class RewardsCfg:
    lift=RewTerm(func=reward,weight=2.,params={'kind':'lift'})
    support=RewTerm(func=reward,weight=2.,params={'kind':'support'})
    balance=RewTerm(func=reward,weight=3.,params={'kind':'balance'})
    success=RewTerm(func=reward,weight=10.,params={'kind':'success'})
    position=RewTerm(func=reward,weight=-.2,params={'kind':'position'})
    rate=RewTerm(func=reward,weight=-.005,params={'kind':'rate'})
    fall=RewTerm(func=reward,weight=-5.,params={'kind':'fall'})
@configclass
class EventsCfg:
    reset_robot=EventTerm(func=reset,mode='reset')
@configclass
class DonesCfg:
    timeout=DoneTerm(func=lab_mdp.time_out,time_out=True)
    failure=DoneTerm(func=failed)
@configclass
class EmptyCfg: pass
@configclass
class UprightEnvCfg(ManagerBasedRLEnvCfg):
    scene=SceneCfg(num_envs=2048,env_spacing=4.)
    actions=ActionsCfg();observations=ObsCfg();rewards=RewardsCfg()
    events=EventsCfg();terminations=DonesCfg();commands=EmptyCfg();curriculum=EmptyCfg()
    def __post_init__(self):
        super().__post_init__()
        self.decimation=10;self.episode_length_s=8.
        self.sim.dt=SPEC['physics_dt'];self.sim.render_interval=self.decimation
        self.sim.physics=PhysxCfg();self.sim.physics.gpu_max_rigid_patch_count=10*2**15
        self.scene.contact_forces.update_period=self.sim.dt
