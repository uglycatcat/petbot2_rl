"""Cycle v8: learned frame/common balance with bounded yaw feedback and residual."""
import torch
from isaaclab.managers import ActionTermCfg, ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
from . import continuous_task as base
from .continuous_task import SPEC, ensure
from .feedback import YawFeedback


class FeedbackAction(base.ContinuousAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.yaw_feedback=YawFeedback(env.num_envs,env.device,env.step_dt,self.max_speed,env.cfg.wheel_slew)

    def process_actions(self,actions):
        previous=self.speed.clone()
        super().process_actions(actions)
        p=ensure(self._env)
        self.speed.copy_(self.yaw_feedback.process(previous,self.speed,p.phase,p.phase_time,p.command,p.motion,actions[:,2]))
        self.applied[:,1:]=self.speed/self.max_speed

    def reset(self,env_ids=None):
        super().reset(env_ids)
        self.yaw_feedback.reset(env_ids)


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
    def __post_init__(self):
        super().__post_init__()
        # A configured update period alone does not advance a lazy sensor's
        # history. Capture every PhysX substep, not only the policy's reads.
        self.scene.lazy_sensor_update=False
        self.scene.contact_forces.update_period=0.
