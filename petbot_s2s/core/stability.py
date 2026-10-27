"""Stability-first reward and persistent, phase-aware loss-of-support detection."""
import torch

STABILITY_CONTRACT = dict(version='persistent_support_v1', front_height_m=.07,
    low_front_seconds=.30, rear_duty=.5, lost_rear_seconds=.30,
    body_contact_seconds=.12, speed_margin_mps=.15, minimum_runaway_mps=.20,
    runaway_seconds=.60, tracking_absolute=[.01,.03], tracking_relative=.5)


def coarse_tracking_score(motion, command):
    tolerance = command.new_tensor([.01,.03])+.5*command.abs()
    excess = ((motion-command).abs()-tolerance).clamp_min(0.)/tolerance
    score = torch.exp(-excess.square())
    # A generous error band must not make standing still optimal for nonzero commands.
    response = (motion*command.sign()/(.5*command.abs()).clamp_min(1e-5)).clamp(0.,1.)
    score *= torch.where(command.abs()>1e-5,response,torch.ones_like(response))
    return .25*score.sum(-1)+.5*score.prod(-1)


class StabilityMonitor:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.dwell=torch.zeros(n,4,device=device)

    def reset(self,ids):
        self.dwell[ids]=0.

    def update(self,phase,state,motion,command,geometry_mode=False):
        eligible=phase==2
        speed_limit=torch.maximum(command[:,0].abs()+.15,command.new_full((len(phase),),.20))
        conditions=torch.stack((state['front_z']<.07,state['rear_fraction']<.5,
            state['body_history'],motion[:,0].abs()>speed_limit),-1)&eligible[:,None]
        if geometry_mode:
            from .kinematics import wheel_geometry
            gap,rear_skew,_=wheel_geometry(state['g'],state['q'][:,3:])
            conditions=torch.stack((gap.amin(-1)<.025,rear_skew>.06,
                state['g'][:,2]>.2,motion[:,0].abs()>speed_limit),-1)&eligible[:,None]
        self.dwell=torch.where(conditions,self.dwell+self.dt,0.)
        thresholds=self.dwell.new_tensor([.30,.30,.12,.60])
        return self.dwell+1e-6>=thresholds
