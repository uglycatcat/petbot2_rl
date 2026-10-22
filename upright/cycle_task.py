"""Full upright cycle with continuous policy control and safe ordinary-drive handoff."""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp as lab_mdp
from .task import UprightEnvCfg, SPEC, POSITION_JOINTS, REAR_WHEELS, FRONT_WHEELS, tensor, state as base_state
from .moving import slew, wheel_targets, projected_heading_rate, physical_contact_deficit
from .cycle import CycleProgress


def state(env):
    s=base_state(env);h=s['heading'];heading=h/h.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    s['yaw']=torch.atan2(heading[:,1],heading[:,0]);s['forward']=(s['vel'][:,:2]*heading).sum(-1)
    x,y,z,w=s['quat'].unbind(-1)
    forward_axis=torch.cat((h,(2*(x*z-w*y))[:,None]),-1)
    s['heading_rate']=projected_heading_rate(forward_axis,s['vel'][:,3:])
    sensor=env.scene['contact_forces']
    sf=env.upright_ids[2];f=tensor(sensor.data.net_forces_w)
    body=[i for i,n in enumerate(sensor.body_names) if n not in (*REAR_WHEELS,*FRONT_WHEELS)]
    s['body_contact']=(f[:,body].norm(dim=-1)>.1).any(-1)
    s['four']=s['rear'] & (f[:,sf,2]>.05).all(-1)
    history=tensor(sensor.data.net_forces_w_history)
    rear=env.upright_ids[3];other=env.upright_ids[4]
    s['rear_fraction']=(history[:,:,rear,2]>.05).all(-1).float().mean(-1)
    s['four_fraction']=((history[:,:,rear,2]>.05).all(-1)&(history[:,:,sf,2]>.05).all(-1)).float().mean(-1)
    s['other_history']=(history[:,:,other].norm(dim=-1)>.1).any(-1).any(-1)
    s['body_history']=(history[:,:,body].norm(dim=-1)>.1).any(-1).any(-1)
    s['front_force_peak']=history[:,:,sf].norm(dim=-1).amax(dim=(1,2))
    s['front_contact']=(history[:,:,sf,2]>.05).any(-1).any(-1)
    s['front_vertical_speed']=tensor(env.scene['robot'].data.body_lin_vel_w)[:,env.upright_ids[0],2].amin(-1)
    return s


def ensure(env):
    if not hasattr(env,'moving_progress'):
        s=state(env);env.moving_progress=CycleProgress(env.num_envs,env.device,env.step_dt,env.cfg.speed_limit,env.cfg.yaw_limit,env.cfg.frame_range,env.cfg.seed,env.cfg.feedback_mode)
        env.moving_progress.reset(torch.arange(env.num_envs,device=env.device),s['pos'],s['yaw'])
        env.moving_step=-1
    return env.moving_progress


class MovingAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.robot=env.scene['robot']
        self.pos_ids=self.robot.find_joints(list(POSITION_JOINTS),preserve_order=True)[0]
        self.wheel_ids=self.robot.find_joints(list(REAR_WHEELS),preserve_order=True)[0]
        self._raw=torch.zeros(env.num_envs,3,device=env.device)
        self.target=tensor(self.robot.data.default_joint_pos)[:,self.pos_ids].clone()
        self.speed=torch.zeros(env.num_envs,2,device=env.device)
        self.lo,self.hi=SPEC['joints']['left_frame']['range']
        self.max_speed=SPEC['actuators']['left_rear_wheel']['control_range'][1]
        self.applied=self._raw.clone();self.previous=self._raw.clone()
    @property
    def action_dim(self):return 3
    @property
    def raw_actions(self):return self._raw
    @property
    def processed_actions(self):return torch.cat((self.target,self.speed),-1)
    def process_actions(self,actions):
        if not torch.isfinite(actions).all() or actions.abs().max()>1.00001:raise ValueError('Invalid moving action')
        p=ensure(self._env);p.before_step();self._raw.copy_(actions);self.previous.copy_(self.applied)
        desired_frame=self.lo+(actions[:,0]+1)*.5*(self.hi-self.lo)
        desired_speed=wheel_targets(actions,self.max_speed,self._env.cfg.wheel_mode)
        # Warmup is ordinary driving, never a reference lift motion. MJCF rear
        # wheel radius .03 m, axle separation .10 m; empirically verify sign/slip.
        drive=torch.stack((p.command[:,0]-.05*p.command[:,1],p.command[:,0]+.05*p.command[:,1]),-1)/.03
        desired_frame=torch.where(p.active,desired_frame,p.frame)
        desired_speed=torch.where(p.active[:,None],desired_speed,drive.clamp(-self.max_speed,self.max_speed))
        ordinary=p.phase>=4
        desired_frame=torch.where(ordinary,p.destination,desired_frame)
        desired_speed=torch.where(ordinary[:,None],drive.clamp(-self.max_speed,self.max_speed),desired_speed)
        frame=slew(self.target[:,3],desired_frame,self._env.cfg.frame_slew*self._env.step_dt)
        self.target[:,3:]=frame[:,None]
        self.speed.copy_(slew(self.speed,desired_speed,self._env.cfg.wheel_slew*self._env.step_dt))
        self.applied[:,0]=2*(frame-self.lo)/(self.hi-self.lo)-1
        self.applied[:,1:]=self.speed/self.max_speed
    def apply_actions(self):
        self.robot.set_joint_position_target(self.target,joint_ids=self.pos_ids)
        self.robot.set_joint_velocity_target(self.speed,joint_ids=self.wheel_ids)
    def reset(self,env_ids=None):
        if env_ids is None:env_ids=slice(None)
        self.target[env_ids]=tensor(self.robot.data.default_joint_pos)[env_ids][:,self.pos_ids]
        self.speed[env_ids]=0;self._raw[env_ids]=0
        self.applied[env_ids]=0;self.applied[env_ids,0]=2*(SPEC['joints']['left_frame']['default']-self.lo)/(self.hi-self.lo)-1
        self.previous[env_ids]=self.applied[env_ids]


@configclass
class MovingActionCfg(ActionTermCfg):
    class_type:type=MovingAction
    asset_name:str='robot'


def metrics(env):
    p=ensure(env)
    if env.moving_step!=env.common_step_counter:
        s=state(env);fixed=s['q'].new_tensor([SPEC['joints'][n]['default'] for n in POSITION_JOINTS[:3]])
        m=p.update(s,fixed);env.moving_metrics=(s,m);env.moving_step=env.common_step_counter
        action=env.action_manager.get_term('moving')
        env.moving_snapshot=({k:v.clone() for k,v in s.items()},{k:v.clone() for k,v in m.items()},p.raw.clone(),p.command.clone(),action.applied.clone(),p.since.clone())
        env.extras.setdefault('log',{}).update({'Cycle/active':p.active.float().mean(),'Cycle/support':m['support'].float().mean(),'Cycle/balance':(p.phase==2).float().mean(),'Cycle/landing':(p.phase==3).float().mean(),'Cycle/complete':m['completed'].float().mean(),'Cycle/qualified':m['qualified'].float().mean()})
    return env.moving_metrics


def set_remote_command(env, command, env_ids=None):
    ensure(env).set_command(command,env_ids)


def observations(env):
    s=state(env);p=ensure(env);a=env.action_manager.get_term('moving')
    base=torch.cat((s['g'],s['heading'],s['vel'],s['q'],s['qd']*.1,s['wheel']*.05,
        p.raw,p.command,s['pos'][:,:2]-p.reference,s['pos'][:,:2]-p.raw_reference,
        torch.stack((p.raw_yaw.sin(),p.raw_yaw.cos()),-1),a.applied,
        p.active[:,None].float(),p.since[:,None]/12,p.peak[:,None],p.support_run[:,None].clamp(max=1),p.frame[:,None]),-1)
    return torch.cat((base,torch.nn.functional.one_hot(p.phase,num_classes=5).float(),p.mode[:,None],p.destination[:,None],p.phase_time[:,None]/4),-1)


def reward(env,kind):
    s,m=metrics(env);a=env.action_manager.get_term('moving')
    if kind in ('entered_balance','entered_handoff'):return m[kind].float()/env.step_dt
    if kind=='failure':return m['failure'].float()/env.step_dt
    if kind=='rate':
        frame_rate=(a.target[:,3]-a.previous[:,0]*(a.hi-a.lo)/2-(a.hi+a.lo)/2)/env.step_dt
        wheel_rate=(a.applied[:,1:]-a.previous[:,1:])*a.max_speed/env.step_dt
        return ((frame_rate/env.cfg.frame_slew).square()+ (wheel_rate/env.cfg.wheel_slew).square().mean(-1))*m['active']
    if kind=='physical_contact':return (1-s['rear_fraction'])*m['active']
    return m[kind].float()


def failed(env):return metrics(env)[1]['failure']
def time_out(env):
    p=ensure(env);m=metrics(env)[1]
    return (m['completed'] | (p.active & (p.since>=env.cfg.active_seconds))) & ~m['failure']


def reset(env,env_ids):
    p=ensure(env);r=env.scene['robot']
    env.extras.setdefault('log',{}).update({'Cycle/episode_balance':p.balance_ever[env_ids].float().mean(),'Cycle/episode_handoff':p.handoff[env_ids].float().mean(),'Cycle/episode_takeover':p.active[env_ids].float().mean()})
    root=tensor(r.data.default_root_state)[env_ids].clone();root[:,:3]+=tensor(env.scene.env_origins)[env_ids];root[:,7:]=0
    r.write_root_state_to_sim(root,env_ids=env_ids)
    q=tensor(r.data.default_joint_pos)[env_ids].clone();r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
    # Reset from written state, never cached articulation pose. Rolling starts later.
    pos=torch.zeros(env.num_envs,3,device=env.device);pos[env_ids]=root[:,:3]-tensor(env.scene.env_origins)[env_ids]
    p.reset(env_ids,pos,torch.zeros(env.num_envs,device=env.device));env.moving_step=-1

@configclass
class ActionsCfg:
    moving=MovingActionCfg()
@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        state=ObsTerm(func=observations)
        def __post_init__(self):
            self.enable_corruption=False;self.concatenate_terms=True;self.history_length=5;self.flatten_history_dim=True
    policy=Policy()
@configclass
class RewardsCfg:
    lift=RewTerm(func=reward,weight=3.,params={'kind':'lift_reward'})
    enter_balance=RewTerm(func=reward,weight=5.,params={'kind':'entered_balance'})
    balance=RewTerm(func=reward,weight=4.,params={'kind':'balance_reward'})
    tracking=RewTerm(func=reward,weight=5.,params={'kind':'track_reward'})
    transition=RewTerm(func=reward,weight=1.,params={'kind':'transition_track'})
    landing=RewTerm(func=reward,weight=3.,params={'kind':'land_reward'})
    approach=RewTerm(func=reward,weight=2.,params={'kind':'approach_reward'})
    enter_handoff=RewTerm(func=reward,weight=10.,params={'kind':'entered_handoff'})
    handoff=RewTerm(func=reward,weight=3.,params={'kind':'handoff_reward'})
    smooth=RewTerm(func=reward,weight=.5,params={'kind':'smooth_reward'})
    rate=RewTerm(func=reward,weight=-.2,params={'kind':'rate'})
    body=RewTerm(func=reward,weight=-3.,params={'kind':'body_cost'})
    touchdown=RewTerm(func=reward,weight=-3.,params={'kind':'touchdown_cost'})
    failure=RewTerm(func=reward,weight=-10.,params={'kind':'failure'})
    contact=RewTerm(func=reward,weight=-2.,params={'kind':'physical_contact'})
@configclass
class EventsCfg:
    reset_robot=EventTerm(func=reset,mode='reset')
@configclass
class DonesCfg:
    timeout=DoneTerm(func=time_out,time_out=True)
    failure=DoneTerm(func=failed)
@configclass
class CycleEnvCfg(UprightEnvCfg):
    actions=ActionsCfg();observations=ObservationsCfg();rewards=RewardsCfg();events=EventsCfg();terminations=DonesCfg()
    feedback_mode:str='command'
    wheel_slew:float=120.
    frame_slew:float=3.
    wheel_mode:str='independent'
    speed_limit:float=.3
    yaw_limit:float=1.
    frame_range:tuple=(-1.,1.2)
    active_seconds:float=16.
    def __post_init__(self):
        super().__post_init__();self.decimation=30;self.sim.render_interval=30;self.episode_length_s=22.;self.scene.contact_forces.history_length=30
