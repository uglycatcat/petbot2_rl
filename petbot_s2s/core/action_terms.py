"""Training control methods vendored from the pinned task. No Isaac dependency.
See provenance.json; math and scheduling are unchanged. A small robot adapter
supplies named-joint targets and world pose/velocity.
"""
import torch
from .moving import slew, wheel_targets
from .cycle import filter_differential
from .continuous import yaw_feedforward
from .feedback import YawFeedback
from .fast_feedback import FastYawFeedback
from .ground_motion import wheel_heading
POSITION_JOINTS = ('head','left_ear','right_ear','left_frame','right_frame')
REAR_WHEELS = ('left_rear_wheel','right_rear_wheel')

def tensor(x): return x

def ensure(env): return env.progress

def observations(env): return env.features()

class ActionTerm:
    def __init__(self,cfg,env): self._env=env

class MovingAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env);self.robot=env.scene['robot']
        self.pos_ids=self.robot.find_joints(list(POSITION_JOINTS),preserve_order=True)[0]
        self.wheel_ids=self.robot.find_joints(list(REAR_WHEELS),preserve_order=True)[0]
        self._raw=torch.zeros(env.num_envs,3,device=env.device)
        self.target=tensor(self.robot.data.default_joint_pos)[:,self.pos_ids].clone()
        self.speed=torch.zeros(env.num_envs,2,device=env.device)
        self.lo,self.hi=env.spec['joints']['left_frame']['range']
        self.max_speed=env.spec['drives']['left_rear_wheel']['control_range'][1]
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
        filtered=filter_differential(self.speed,desired_speed,self._env.step_dt,self._env.cfg.yaw_filter_tau,self.max_speed)
        desired_speed=torch.where((p.active&~ordinary)[:,None],filtered,desired_speed)
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
        self.applied[env_ids]=0;self.applied[env_ids,0]=2*(self._env.spec['joints']['left_frame']['default']-self.lo)/(self.hi-self.lo)-1
        self.previous[env_ids]=self.applied[env_ids]

class ContinuousAction(MovingAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.drive_integral = torch.zeros(env.num_envs, 2, device=env.device)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all() or actions.abs().max()>1.00001:
            raise ValueError('Invalid continuous action')
        p = ensure(self._env)
        self._raw.copy_(actions)
        self.previous.copy_(self.applied)
        desired_frame = self.lo+(actions[:, 0]+1)*.5*(self.hi-self.lo)
        desired_speed = wheel_targets(actions, self.max_speed, 'common_diff')
        if self._env.cfg.steering_design == 'residual':
            blend = torch.where(p.phase==2, (p.phase_time/.6).clamp(0, 1),
                    torch.where(p.phase==3, (1.-p.phase_time/.6).clamp(0, 1), 0.))
            difference = (1.-blend)*actions[:, 2]*5.+blend*(yaw_feedforward(p.command[:, 1])+actions[:, 2]*1.5)
            common = actions[:, 1]*self.max_speed
            desired_speed = torch.stack((common-difference, common+difference), -1).clamp(-self.max_speed, self.max_speed)
        if getattr(self._env.cfg,'capability_version','legacy')=='v25':
            from .capability import balance_common
            common=balance_common(actions[:,1]);difference=actions[:,2]*5.
            expanded=torch.stack((common-difference,common+difference),-1).clamp(-self.max_speed,self.max_speed)
            desired_speed=torch.where((p.phase==2)[:,None],expanded,desired_speed)
        ordinary = ~p.active | (p.phase>=4)
        error = p.command-p.motion
        gains = error.new_tensor([20., 8.])
        limit = error.new_tensor([5., 2.])
        integral = self.drive_integral+self._env.step_dt*gains*error
        self.drive_integral[:] = torch.where(ordinary[:, None], torch.maximum(torch.minimum(integral, limit), -limit), 0.)
        common = p.command[:, 0]/.03+10.*error[:, 0]+self.drive_integral[:, 0]
        difference = yaw_feedforward(p.command[:, 1])+2.*error[:, 1]+self.drive_integral[:, 1]
        drive = torch.stack((common-difference, common+difference), -1).clamp(-self.max_speed, self.max_speed)
        desired_speed = torch.where(ordinary[:, None], drive, desired_speed)
        desired_frame = torch.where(~p.active, p.frame, desired_frame)
        desired_frame = torch.where(p.phase>=4, p.destination, desired_frame)
        filtered = filter_differential(self.speed, desired_speed, self._env.step_dt, self._env.cfg.yaw_filter_tau, self.max_speed)
        desired_speed = torch.where(ordinary[:, None], desired_speed, filtered)
        frame = slew(self.target[:, 3], desired_frame, self._env.cfg.frame_slew*self._env.step_dt)
        self.target[:, 3:] = frame[:, None]
        self.speed.copy_(slew(self.speed, desired_speed, self._env.cfg.wheel_slew*self._env.step_dt))
        self.applied[:, 0] = 2*(frame-self.lo)/(self.hi-self.lo)-1
        self.applied[:, 1:] = self.speed/self.max_speed

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.drive_integral[slice(None) if env_ids is None else env_ids] = 0.

class FeedbackAction(ContinuousAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.yaw_feedback=YawFeedback(env.num_envs,env.device,env.step_dt,self.max_speed,env.cfg.wheel_slew)
        period=.002 if env.cfg.yaw_feedback_design=='fast2' else .02
        self.fast_feedback=FastYawFeedback(env.num_envs,env.device,self.max_speed,env.step_dt,env.cfg.wheel_slew,period,env.cfg.yaw_feedback_gain)
        self.fast_stride=round(period/env.physics_dt)
        assert self.fast_stride>=1 and abs(self.fast_stride*env.physics_dt-period)<1e-9
        self.fast_enabled=env.cfg.yaw_feedback_design in ('fast20','fast2')
        self.inner_index=0
        self.inner_mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.inner_speed=self.speed.clone();self.inner_policy=self.speed.clone();self.inner_anchor=self.speed.clone()
        self.continuous_heading=getattr(env.cfg,'transition_contract','legacy')=='linear_heading'
        if self.continuous_heading:self.fast_feedback.half_difference_limit=self.max_speed
        self.full_range=getattr(env.cfg,'task_profile','legacy')=='full_range'
        self.fast_feedback.anti_windup=self.full_range
        self.source_ordinary=getattr(env.cfg,'capability_version','legacy')=='v25'
        if self.source_ordinary:
            from .ordinary import OrdinaryDrive
            self.ordinary_drive=OrdinaryDrive(env.num_envs,env.device,self.max_speed)

    def process_actions(self,actions):
        previous=self.speed.clone()
        old_integral=self.yaw_feedback.integral.clone()
        super().process_actions(actions)
        p=ensure(self._env)
        if not self.continuous_heading:
            self.speed.copy_(self.yaw_feedback.process(previous,self.speed,p.phase,p.phase_time,p.command,p.motion,actions[:,2]))
        self.applied[:,1:]=self.speed/self.max_speed
        if self.fast_enabled:
            self.inner_index=0
            self.inner_mask=(p.phase==2)&(p.phase_time>=1.2)
            if self.continuous_heading:
                self.inner_mask=(p.phase>=1)&(p.phase<=3)
                old_integral[~self.inner_mask]=0.
            self.inner_anchor.copy_(previous);self.inner_speed.copy_(previous);self.inner_policy.copy_(self.speed)
            self.fast_feedback.integral.copy_(old_integral)
            self.yaw_feedback.integral[self.inner_mask]=old_integral[self.inner_mask]

    def apply_actions(self):
        if self.fast_enabled and self.inner_index%self.fast_stride==0:
            from .ground_motion import wheel_heading
            p=ensure(self._env);d=self.robot.data
            _,heading,rate=wheel_heading(tensor(d.root_link_pose_w)[:,3:7],tensor(d.root_com_vel_w))
            command=p.command;residual=self.raw_actions[:,2]
            if self.continuous_heading:
                from .moving import wrap
                transition=(p.phase==1)|(p.phase==3)
                command=command.clone()
                command[transition,1]=(3.*wrap(p.transition_heading-heading))[transition].clamp(-15. if self.full_range else -1.5,15. if self.full_range else 1.5)
                residual=torch.where(transition,0.,residual)
            candidate=self.fast_feedback.process(self.inner_speed,self.inner_policy,self.inner_anchor,
                command,rate,residual,self.inner_mask)
            self.inner_speed.copy_(candidate);self.speed.copy_(candidate)
            self.applied[:,1:]=self.speed/self.max_speed
            self.yaw_feedback.integral[self.inner_mask]=self.fast_feedback.integral[self.inner_mask]
            self.yaw_feedback.target[self.inner_mask]=self.fast_feedback.target[self.inner_mask]
        if self.source_ordinary:
            from .ground_motion import wheel_heading
            p=ensure(self._env);ordinary=~p.active;d=self.robot.data
            velocity=tensor(d.root_com_vel_w)
            ground,_,rate=wheel_heading(tensor(d.root_link_pose_w)[:,3:7],velocity)
            measured=torch.stack(((velocity[:,:2]*ground).sum(-1),rate),-1)
            wheel=self.ordinary_drive.wheels(p.command,measured,self._env.physics_dt,ordinary)
            self.speed[ordinary]=wheel[ordinary];self.applied[ordinary,1:]=wheel[ordinary]/self.max_speed
        super().apply_actions()
        self.inner_index+=1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.yaw_feedback.reset(env_ids)
        self.fast_feedback.reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        if self.source_ordinary:self.ordinary_drive.reset(ids)
        self.inner_mask[ids]=False
        self.inner_speed[ids]=0.;self.inner_policy[ids]=0.;self.inner_anchor[ids]=0.

def attach_balance_reflex(env,actor,enabled,*,full_parent=False):
    """Install fixed inner feedback without changing the rollout action density."""
    if not enabled:return
    a=env.action_manager.get_term('moving');fixed=actor.parent_balance if full_parent else actor.balance_reflex
    assert env.step_dt==.06 and env.physics_dt==.002 and a.fast_stride==1
    assert not any(p.requires_grad for p in fixed.parameters())
    process=a.process_actions;apply=a.apply_actions;reset=a.reset
    cache={}

    def process_actions(actions):
        raw=a.policy_observation
        assert raw.shape==(env.num_envs,290)
        cache['raw']=raw.clone();cache['frame_anchor']=a.target[:,3].clone()
        with torch.no_grad():
            if full_parent:
                parent=fixed(raw)[:,:2]
            else:
                base=fixed.base(raw);parent=(base+fixed(raw)).clamp(-.99,.99)
                cache['base']=base
            cache['offset']=actions[:,:2].detach()-parent
        process(actions)

    def apply_actions():
        p=ensure(env);mask=(p.phase==2)&(p.phase_time>=1.8)
        if a.inner_index in (10,20) and mask.any():
            raw=cache['raw'].clone();raw[:,-58:]=observations(env)
            with torch.no_grad():
                mean=fixed(raw)[:,:2] if full_parent else cache['base']+fixed(raw)
                action=(mean+cache['offset']).clamp(-.99,.99)
            previous=a.inner_policy.mean(-1);desired=action[:,1]*a.max_speed
            if getattr(env.cfg,'capability_version','legacy')=='v25':
                from .capability import balance_common
                desired=balance_common(action[:,1])
            common=previous+(desired-previous).clamp(-env.cfg.wheel_slew*.02,env.cfg.wheel_slew*.02)
            bound=env.cfg.wheel_slew*env.step_dt;anchor=a.inner_anchor
            lower=(anchor-bound).clamp_min(-a.max_speed).mean(-1)
            upper=(anchor+bound).clamp_max(a.max_speed).mean(-1)
            common=common.clamp(min=lower,max=upper)
            a.inner_policy[mask]=common[mask,None]
            desired=a.lo+(action[:,0]+1)*.5*(a.hi-a.lo)
            previous=a.target[:,3];limit=env.cfg.frame_slew*.02
            frame=previous+(desired-previous).clamp(-limit,limit)
            bound=env.cfg.frame_slew*env.step_dt
            frame=frame.clamp(min=cache['frame_anchor']-bound,max=cache['frame_anchor']+bound).clamp(a.lo,a.hi)
            a.target[mask,3:]=frame[mask,None]
            a.applied[:,0]=2*(a.target[:,3]-a.lo)/(a.hi-a.lo)-1
        apply()

    def reset_actions(env_ids=None):
        reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        for value in cache.values():value[ids]=0.

    a.process_actions=process_actions;a.apply_actions=apply_actions;a.reset=reset_actions
