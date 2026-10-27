"""Bounded ground-yaw feedback; learned frame/common targets stay at 60 ms."""
import torch
from .continuous import yaw_feedforward

FAST_CONTROLLER=dict(period=.02,kp=8.,ki=4.,integral_limit=1.5,half_difference_limit=2.,
                     difference_rate=12.,residual_scale=.25,active_after=1.2,
                     measurement='instant_wheel_lateral_heading_rate')


class FastYawFeedback:
    def __init__(self,n,device,max_speed=25.,control_dt=.06,wheel_slew=480.,period=.02,kp=8.):
        assert period in (.02,.002)
        assert kp in (8.,16.)
        self.period=period
        self.kp=kp
        self.half_difference_limit=2.
        self.max_speed=max_speed;self.control_limit=control_dt*wheel_slew
        self.integral=torch.zeros(n,device=device);self.target=torch.zeros_like(self.integral)

    def reset(self,ids=None):
        if ids is None:ids=slice(None)
        self.integral[ids]=0.;self.target[ids]=0.

    def process(self,previous,policy,anchor,command,rate,residual,active):
        error=command[:,1]-rate
        integral=(self.integral+self.period*4.*error).clamp(-1.5,1.5)
        self.integral[active]=integral[active]
        desired=(yaw_feedforward(command[:,1])+self.kp*error+self.integral+.25*residual).clamp(-self.half_difference_limit,self.half_difference_limit)
        old=(previous[:,1]-previous[:,0])*.5
        half=old+(desired-old).clamp(-12.*self.period,12.*self.period)
        common=policy.mean(-1)
        margin=(self.max_speed-common.abs()).clamp_min(0.)
        # Retain the learned common target exactly, within both wheel speed and
        # whole control-interval slew limits. Inner feedback cannot bypass them.
        lower=torch.stack((-margin,common-anchor[:,0]-self.control_limit,
                           anchor[:,1]-common-self.control_limit),-1).amax(-1)
        upper=torch.stack((margin,common-anchor[:,0]+self.control_limit,
                           anchor[:,1]-common+self.control_limit),-1).amin(-1)
        half=torch.maximum(torch.minimum(half,upper),lower)
        output=torch.stack((common-half,common+half),-1)
        self.target[active]=half[active]
        return torch.where(active[:,None],output,policy)
