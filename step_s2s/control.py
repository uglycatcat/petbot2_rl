"""Identical common targets, asymmetric hip scaling and per-physics-step slew."""
import numpy as np


class Controller:
    def __init__(self,training):
        self.training=training;self.mapping=training.get('wheel_mapping','affine')
        if self.mapping not in ('affine','zero_centered','zero_margin'):raise ValueError('Unknown wheel mapping')
        self.reset()

    def reset(self):
        self.action=np.array([0.,-1.],np.float32);self.hip=np.float32(0.);self.wheel=np.float32(0.)

    @property
    def executed(self):
        return np.array([self.hip/np.float32(1.2 if self.hip>=0 else 1.),self.wheel/np.float32(12.5)-np.float32(1)],np.float32)

    def request(self,action):
        a=np.asarray(action,np.float32)
        if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError('Policy must return two finite actions')
        self.action=np.clip(a,-1.,1.)

    def advance(self,dt):
        h=self.action[0]*np.float32(1.2 if self.action[0]>=0 else 1.)
        if self.mapping=='affine':w=(self.action[1]+np.float32(1.))*np.float32(12.5)
        elif self.mapping=='zero_centered':w=max(self.action[1],np.float32(0.))*np.float32(25.)
        else:w=np.clip((self.action[1]+np.float32(.9))/np.float32(1.9),0.,1.)*np.float32(25.)
        dh=np.float32(self.training['hip_target_rate_rad_s']*dt);dw=np.float32(self.training['wheel_target_accel_rad_s2']*dt)
        self.hip=np.float32(self.hip+np.clip(h-self.hip,-dh,dh))
        self.wheel=np.float32(self.wheel+np.clip(w-self.wheel,-dw,dw))
        return self.hip,self.wheel
