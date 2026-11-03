"""Physical-substep single-flight diagnostics; never a policy input."""
import numpy as np


class FrontMotion:
    def __init__(self):self.reset()

    def reset(self):
        self.air=np.zeros(2,bool);self.count=np.zeros(2,np.int64)
        self.off_time=np.zeros(2,np.float32);self.floor_time=np.zeros(2,np.float32);self.top_time=np.zeros(2,np.float32)
        self.returned_floor=np.zeros(2,bool);self.landed_top=np.zeros(2,bool)

    def update(self,minimum,force,platform_normal,edge_x,height,dt):
        z,x=minimum[:,2],minimum[:,0];over_top=x>edge_x+.0005
        plane=np.where(over_top,height,0.);magnitude=np.linalg.norm(force,axis=-1)
        off=(z-plane>.004)&(magnitude<.02)
        floor=(z<=.002)&(magnitude>.02)
        top=over_top&(z>=height-.002)&(z<=height+.004)&(platform_normal[:,2]>.02)
        for name,mask in [('off_time',off),('floor_time',floor),('top_time',top)]:
            setattr(self,name,np.where(mask,getattr(self,name)+np.float32(dt),np.float32(0.)))
        leaving=~self.air&(self.off_time>=.02-1e-7)
        self.count+=leaving;self.air|=leaving
        floor_landing=self.air&(self.floor_time>=.02-1e-7)
        top_landing=self.air&(self.top_time>=.02-1e-7)
        self.returned_floor|=floor_landing;self.landed_top|=top_landing
        self.air&=~(floor_landing|top_landing)

    @property
    def invalid(self):return bool((self.returned_floor|(self.count>1)).any())

    @property
    def complete(self):return bool(((self.count==1)&self.landed_top&~self.air).all() and not self.invalid)
