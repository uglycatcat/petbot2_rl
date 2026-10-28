"""Cycle v8: learned frame/common balance with bounded yaw feedback and residual."""
import torch
from isaaclab.managers import ActionTermCfg, ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
from . import continuous_task as base
from .continuous_task import SPEC, ensure
from .feedback import YawFeedback
from .fast_feedback import FastYawFeedback


class FeedbackAction(base.ContinuousAction):
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
            from .cycle_task import tensor
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
        super().apply_actions()
        self.inner_index+=1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.yaw_feedback.reset(env_ids)
        self.fast_feedback.reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self.inner_mask[ids]=False
        self.inner_speed[ids]=0.;self.inner_policy[ids]=0.;self.inner_anchor[ids]=0.


def observations(env):
    old=base.observations(env)
    integral=env.action_manager.get_term('moving').yaw_feedback.integral[:,None]/1.5
    return torch.cat((old,integral),-1)


@configclass
class ActionsCfg:
    moving=ActionTermCfg(class_type=FeedbackAction,asset_name='robot')


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        state=ObsTerm(func=observations)
        def __post_init__(self):
            self.enable_corruption=False;self.concatenate_terms=True
            self.history_length=5;self.flatten_history_dim=True
    policy=Policy()


@configclass
class FeedbackEnvCfg(base.ContinuousEnvCfg):
    actions=ActionsCfg();observations=ObservationsCfg()
    steering_design:str='feedback'
    heading_reference:str='wheel_lateral'
    instant_tracking_weight:float=.3
    yaw_feedback_design:str='legacy60'
    yaw_feedback_gain:float=8.
    tracking_objective:str='mixed'
    transition_contract:str='legacy'
    rise_speed_reference_mode:str='capture'
    def __post_init__(self):
        super().__post_init__()
        # A configured update period alone does not advance a lazy sensor's
        # history. Capture every PhysX substep, not only the policy's reads.
        self.scene.lazy_sensor_update=False
        self.scene.contact_forces.update_period=0.
