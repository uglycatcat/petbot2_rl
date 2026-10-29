"""Source-envelope ordinary PI at physics cadence, matching the MuJoCo port.

Used before takeover only. Landing and phase-4 handoff retain their validated
controller/timing; only their hip destination is reset by the v25 progress.
"""
import torch
from .continuous import yaw_feedforward
from .full_range import project_command

class OrdinaryDrive:
    def __init__(self,n,device,wheel_limit=25.):
        self.wheel_limit=wheel_limit
        self.integral=torch.zeros(n,2,device=device)
    def wheels(self,command,motion,dt,mask):
        command=project_command(command,self.wheel_limit)
        error=command-motion
        candidate=(self.integral+dt*error*error.new_tensor([20.,8.])).clamp(
            min=error.new_tensor([-5.,-2.]),max=error.new_tensor([5.,2.]))
        common=command[:,0]/.03+10.*error[:,0]+candidate[:,0]
        difference=yaw_feedforward(command[:,1])+2.*error[:,1]+candidate[:,1]
        desired=torch.stack((common-difference,common+difference),-1)
        integrate=mask&(desired.abs().amax(-1)<=self.wheel_limit)
        self.integral[integrate]=candidate[integrate];self.integral[~mask]=0.
        return desired.clamp(-self.wheel_limit,self.wheel_limit)
    def reset(self,ids):self.integral[ids]=0.
