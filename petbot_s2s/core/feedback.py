"""Bounded two-wheel yaw feedback, with a small learned velocity correction."""
import torch
from .continuous import yaw_feedforward

CONTROLLER=dict(kp=2.,ki=4.,integral_limit=1.5,half_difference_limit=2.,
                difference_rate=4.,residual_scale=.25,start_seconds=.6,blend_seconds=.6)


class YawFeedback:
    def __init__(self,n,device,dt,max_speed,wheel_slew=480.):
        self.dt=dt;self.max_speed=max_speed;self.wheel_slew=wheel_slew
        self.integral=torch.zeros(n,device=device)
        self.target=torch.zeros(n,device=device)

    def reset(self,ids=None):
        if ids is None:ids=slice(None)
        self.integral[ids]=0.;self.target[ids]=0.

    def process(self,previous,policy_speed,phase,clock,command,motion,residual):
        active=(phase==2)&(clock>=.6)
        self.integral[phase<=1]=0.
        error=command[:,1]-motion[:,1]
        self.integral[active]=(self.integral+4.*self.dt*error)[active].clamp(-1.5,1.5)
        desired=(yaw_feedforward(command[:,1])+2.*error+self.integral+.25*residual).clamp(-2.,2.)
        old=(previous[:,1]-previous[:,0])*.5
        self.target=old+(desired-old).clamp(-4.*self.dt,4.*self.dt)
        blend=torch.where(phase==2,((clock-.6)/.6).clamp(0,1),torch.where(phase==3,(1-clock/.6).clamp(0,1),0.))
        policy_half=(policy_speed[:,1]-policy_speed[:,0])*.5
        difference=policy_half*(1-blend)+self.target*blend
        common=policy_speed.mean(-1);margin=(self.max_speed-common.abs()).clamp_min(0)
        difference=torch.maximum(torch.minimum(difference,margin),-margin)
        output=policy_speed.clone();mask=blend>0
        output[mask]=torch.stack((common-difference,common+difference),-1)[mask]
        delta=output-previous;limit=self.wheel_slew*self.dt
        return torch.where(delta.abs()>limit,previous+delta.clamp(-limit,limit),output)
