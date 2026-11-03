"""Truth-only diagnostics. These values never enter the actor or estimator."""
import numpy as np
from .front_motion import FrontMotion

WHEELS=('left_front_wheel','right_front_wheel','left_rear_wheel','right_rear_wheel')


def terrain_clearance(points,edges,edge,height,length,half_width):
    x=points[:,0]-edge;z=points[:,2]
    floor=max(0.,z.min());dx=np.maximum(np.maximum(-x,x-length),0);dz=np.maximum(np.maximum(-z,z-height),0)
    d2=np.min(dx*dx+dz*dz);ax=x[edges[:,0]];az=z[edges[:,0]]
    vx=x[edges[:,1]]-ax;vz=z[edges[:,1]]-az;norm=np.maximum(vx*vx+vz*vz,1e-20)
    for cx in (0.,length):
        t=np.clip(((cx-ax)*vx+(height-az)*vz)/norm,0,1)
        d2=min(d2,np.min((ax+t*vx-cx)**2+(az+t*vz-height)**2))
    ix=np.where(abs(vx)>1e-12,vx,1.);iz=np.where(abs(vz)>1e-12,vz,1.)
    tx0=-ax/ix;tx1=(length-ax)/ix;tz0=-az/iz;tz1=(height-az)/iz
    lx=np.where(abs(vx)>1e-12,np.minimum(tx0,tx1),0);hx=np.where(abs(vx)>1e-12,np.maximum(tx0,tx1),1)
    lz=np.where(abs(vz)>1e-12,np.minimum(tz0,tz1),0);hz=np.where(abs(vz)>1e-12,np.maximum(tz0,tz1),1)
    hit=np.maximum(np.maximum(lx,lz),0)<=np.minimum(np.minimum(hx,hz),1)
    hit&=(abs(vx)>1e-12)|((ax>=0)&(ax<=length));hit&=(abs(vz)>1e-12)|((az>=0)&(az<=height))
    if hit.any():d2=0.
    lateral=max(points[:,1].min()-half_width,-half_width-points[:,1].max(),0.)
    return max(0.,min(floor,np.sqrt(d2+lateral*lateral))-.0001)


class Monitor:
    def __init__(self,metadata,edge):
        self.task=metadata['task'];self.edge=edge
        self.single_lift=metadata['training'].get('reward_version') in ('clean_v17','pitch_v18')
        self.front_motion=FrontMotion();self.reset()

    def reset(self):
        self.front_motion.reset()
        self.lifted=np.zeros(2,bool);self.cleared=np.zeros(2,bool);self.landed=np.zeros(2,bool)
        self.contact_age=np.ones(4);self.supported=np.zeros(4,bool);self.rear_air=np.zeros(2)
        self.failure=0;self.phase=0;self.drive_start=None;self.gap=0.;self.stop_time=0.;self.completed=False
        self.ever_fourtop=False;self.ever_home=False;self.ever_drive=False;self.peak_wheel_speed=0.;self.peak_contact=0.

    def capture(self,minimum,maximum,forces,platform_force,contact_x,clearance,com,up_z,body_hit,wheel_speed,dt):
        h=self.task['terrain']['height_m'];above=minimum[:2,2]>=h+.0005;before=maximum[:2,0]<self.edge-.0005
        self.lifted|=before&above
        self.cleared|=self.lifted&(minimum[:2,0]>self.edge+.0005)&above
        top=(platform_force[:,2]>.02)&(minimum[:,2]>=h-.002)&(minimum[:,0]>self.edge+.0005)
        bad=((np.linalg.norm(platform_force[:2,:2],axis=-1)>.01)&(minimum[:2,0]<=self.edge+.0005))
        bad|=(np.linalg.norm(platform_force[:2],axis=-1)>.02)&(contact_x[:2]<=self.edge+.0005)
        self.landed|=self.cleared&top[:2]
        if self.single_lift:self.front_motion.update(minimum[:2],forces[:2],platform_force[:2],self.edge,h,dt)
        self.contact_age=np.where(top,0.,self.contact_age+dt)
        self.supported=(self.contact_age<=.02+1e-12)&(minimum[:,2]>=h-.002)&(minimum[:,2]<=h+.004)&(minimum[:,0]>self.edge+.0005)
        rear=(clearance>.004)&(np.linalg.norm(forces[2:],axis=-1)<.02)
        self.rear_air=np.where(rear,self.rear_air+dt,0.)
        code=1 if bad.any() else 0
        if (self.rear_air>.02).any():code=2
        if up_z<.4 or body_hit:code=3
        if com[0]<-.05 or abs(com[1])>.15:code=4
        if not code and self.single_lift and self.front_motion.invalid:code=5
        if not self.failure:self.failure=code
        self.peak_wheel_speed=max(self.peak_wheel_speed,float(np.max(np.abs(wheel_speed))))
        self.peak_contact=max(self.peak_contact,float(np.linalg.norm(forces,axis=-1).max()))

    def control_step(self,minimum,maximum,com,velocity,yaw,yaw_rate,hips,dt):
        a=self.task['acceptance'];t=self.task['terrain']
        aligned=abs(com[1])<=a['line_tolerance_m'] and abs(yaw)<=a['heading_tolerance_rad']
        pose=bool(self.landed.all() and aligned and np.max(np.abs(hips))<=a['hip_tolerance_rad'] and not self.failure)
        ready=bool(self.supported.all() and pose)
        if self.single_lift:ready=ready and self.front_motion.complete
        near=bool(((minimum[:,2]>=t['height_m']-.003)&(minimum[:,2]<=t['height_m']+.004)
              &(minimum[:,0]>self.edge+.0005)&(maximum[:,0]<self.edge+t['platform_length_m'])
              &(np.abs(minimum[:,1])<t['platform_width_m']/2)&(np.abs(maximum[:,1])<t['platform_width_m']/2)).all())
        self.gap=0. if ready else self.gap+dt
        if ready and self.drive_start is None:self.drive_start=float(com[0])
        if not (ready or (pose and near and self.gap<=.06+1e-7)):self.drive_start=None
        drove=ready and self.drive_start is not None and com[0]-self.drive_start>=a['post_climb_distance_m']
        stopped=drove and np.linalg.norm(velocity)<=a['stop_speed_m_s'] and abs(yaw_rate)<=a['stop_yaw_rate_rad_s']
        self.stop_time=self.stop_time+dt if stopped else 0.
        self.completed=self.stop_time+1e-7>=a['stop_hold_s'] and not self.failure
        self.phase=4 if self.completed else 3 if drove else 2 if ready else 1 if self.landed.all() else 0
        self.ever_fourtop|=bool(self.supported.all() and self.landed.all() and not self.failure)
        self.ever_home|=ready;self.ever_drive|=bool(drove)
