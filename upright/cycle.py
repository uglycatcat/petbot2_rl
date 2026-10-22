"""Full moving upright cycle: rise, balance, land, then continuous ordinary-drive handoff."""
import torch
from .moving import MovingProgress, wrap


class CycleProgress(MovingProgress):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        z=lambda:torch.zeros(self.n,device=self.device)
        self.phase=z().long();self.mode=z()+1;self.destination=z()
        self.phase_time=z();self.land_at=z();self.ground_run=z();self.balance_time=z()
        self.rise_position=torch.zeros(self.n,2,device=self.device)
        self.transition_offset=z();self.front_previous=z()+.045
        self.max_impact=z();self.max_touchdown_speed=z();self.last_front_speed=z();self.landed=z().bool();self.handoff=z().bool()
        self.balance_ever=z().bool();self.external_mode=z().bool();self.land_height=z()+.18
        self.track_sum=torch.zeros(self.n,2,device=self.device);self.track_count=z()
        self.track_good=z();self.balance_good=z();self.handoff_run=z();self.landing_potential=z()

    def reset(self,ids,pos,heading):
        super().reset(ids,pos,heading)
        for x in (self.phase,self.phase_time,self.ground_run,self.balance_time,self.transition_offset,
                  self.max_impact,self.max_touchdown_speed,self.last_front_speed,self.landed,self.handoff,self.balance_ever,self.external_mode,self.track_count,self.track_good,self.balance_good,self.handoff_run,self.landing_potential):x[ids]=0
        self.track_sum[ids]=0;self.mode[ids]=1
        self.destination[ids]=self.frame_range[0]+self.uniform(ids,8)*(self.frame_range[1]-self.frame_range[0])
        self.land_at[ids]=5.+3.*self.uniform(ids,9)
        self.rise_position[ids]=pos[ids,:2];self.front_previous[ids]=.045;self.land_height[ids]=.18

    def request_mode(self,mode,frame,ids=None):
        if ids is None:ids=torch.arange(self.n,device=self.device)
        mode=torch.as_tensor(mode,device=self.device,dtype=self.mode.dtype).expand(len(ids))
        frame=torch.as_tensor(frame,device=self.device,dtype=self.destination.dtype).expand(len(ids))
        if not ((mode==0)|(mode==1)).all() or not torch.isfinite(frame).all():raise ValueError('Invalid mode request')
        if (frame<self.frame_range[0]).any() or (frame>self.frame_range[1]).any():raise ValueError('Driving frame outside envelope')
        # A requested landing is latched until stable handoff. Chattering cannot reset its deadline/reference.
        accept=(self.phase[ids]<3)
        selected=ids[accept];self.mode[selected]=mode[accept];self.destination[selected]=frame[accept]
        self.external_mode[ids]=True

    def update(self,s,fixed):
        # Base progress supplies warmup admission and command/filter machinery;
        # none of its reward/success definitions are used by this version.
        previous_support=self.support_run.clone();m=super().update(s,fixed)
        switch=m['switched'];self.phase[switch]=1;self.phase_time[switch]=0
        self.rise_position[switch]=s['pos'][switch,:2]
        self.phase_time+=self.active*self.dt
        support=self.active & (s['rear_fraction']>=.9) & ~s['other_history'] & (s['front_z']>.12) & (s['g'][:,2]<-.25)
        self.support_run=torch.where(support,previous_support+self.dt,0.)
        entering=(self.phase==1)&(self.support_run>=.30)
        self.phase[entering]=2;self.phase_time[entering]=0
        self.transition_offset[entering]=(s['pos'][entering,:2]-self.raw_reference[entering]).norm(dim=-1)
        # Exactly once: no debt to the hypothetical pre-rise path. Keep an audit of the offset.
        for ref in (self.reference,self.raw_reference):ref[entering]=s['pos'][entering,:2]
        for yaw in (self.yaw,self.raw_yaw):yaw[entering]=s['yaw'][entering]
        self.balance_ever |= entering
        self.balance_time+=((self.phase==2)&support)*self.dt
        automatic=(self.phase==2)&~self.external_mode&(self.since>=self.land_at)
        self.mode[automatic]=0
        landing=(self.phase==2)&(self.mode==0)
        self.phase[landing]=3;self.phase_time[landing]=0;self.land_height[landing]=s['front_z'][landing].clamp_min(.12)
        ground=(s['four_fraction']>=.9)&~s['body_history']&(s['g'][:,2]<-.85)&(s['g'][:,1].abs()<.15)
        frame_error=(s['q'][:,3:]-self.destination[:,None]).abs().amax(-1)
        gentle=(s['front_vertical_speed'].abs()<.25)&(s['vel'][:,3:5].norm(dim=-1)<1.)
        self.ground_run=torch.where((self.phase==3)&ground&gentle&(frame_error<.1),self.ground_run+self.dt,0.)
        handoff=(self.phase==3)&(self.ground_run>=.30)
        self.phase[handoff]=4;self.phase_time[handoff]=0;self.landed |= handoff
        self.handoff_run=torch.where((self.phase==4)&ground&gentle&(frame_error<.1),self.handoff_run+self.dt,0.)
        self.handoff |= (self.phase==4)&(self.handoff_run>=1.)
        error=torch.stack((s['forward']-self.raw[:,0],s['heading_rate']-self.raw[:,1]),-1)
        # Relative/absolute tolerances keep small nonzero commands accountable.
        tolerance=torch.stack((.005+.25*self.raw[:,0].abs(),.015+.25*self.raw[:,1].abs()),-1)
        tracking=(error.abs()<tolerance).all(-1)
        balanced=support&(m['tilt_rate']<.8)&(s['g'][:,1].abs()<.15)&(m['fixed_error']<.05)
        self.track_sum+=error.abs()*((self.phase==2)&(self.phase_time>.5))[:,None]
        eligible=(self.phase==2)&(self.phase_time>.5)
        self.track_count+=eligible;self.track_good+=eligible&tracking;self.balance_good+=eligible&balanced
        tracking_reward=.5*torch.exp(-(error[:,0]/(2*tolerance[:,0])).square())+.5*torch.exp(-(error[:,1]/(2*tolerance[:,1])).square())
        remaining=((s['front_z']-.045)/(self.land_height-.045).clamp_min(.075)).clamp(0,1)
        lowering=(1-remaining)*s['rear_fraction']
        pitch_return=(( -s['g'][:,2]-.25)/.75).clamp(0,1)
        frame_match=torch.exp(-frame_error.square()/.15**2)
        potential=lowering+pitch_return+frame_match
        # Reward change, never repeated residence at an already lowered pose.
        self.landing_potential[landing]=potential[landing]
        landing_progress=torch.where((self.phase==3)|handoff,potential-self.landing_potential,0.)
        self.landing_potential.copy_(potential)
        clearance=(s['front_z']-.045).clamp_min(0.)
        allowed_down=(.08+torch.sqrt(2*.6*clearance)).clamp(max=.6)
        descent_cost=torch.relu(-s['front_vertical_speed']-allowed_down).square()/.25**2
        # Safety termination has priority over time limits, including the final handoff step.
        failure=m['physical_failure']|m['warmup_failed']|((self.phase==1)&(self.phase_time>3.))|((self.phase==3)&(self.phase_time>4.))
        failure |= self.active & ((s['pos'][:,:2]-self.rise_position).norm(dim=-1)>3.)
        self.max_impact=torch.maximum(self.max_impact,torch.where((self.phase==3),s['front_force_peak'],0.))
        touchdown_speed=torch.maximum((-s['front_vertical_speed']).clamp_min(0),(-self.last_front_speed).clamp_min(0))
        self.max_touchdown_speed=torch.maximum(self.max_touchdown_speed,torch.where((self.phase==3)&s['front_contact'],touchdown_speed,0.))
        self.last_front_speed.copy_(s['front_vertical_speed'])
        result=dict(m)
        result.update(phase=self.phase.clone(),phase_time=self.phase_time.clone(),support=support,
            entered_balance=entering,entered_landing=landing,entered_handoff=handoff,
            failure=failure,balanced=balanced,tracking_good=tracking,
            completed=self.handoff & ~failure,raw_error=error,transition_offset=self.transition_offset.clone(),max_touchdown_speed=self.max_touchdown_speed.clone(),
            qualified=self.handoff & ~failure & (self.track_count>=20) & (self.track_good/self.track_count.clamp_min(1)>=.8) & (self.balance_good/self.track_count.clamp_min(1)>=.9) & (self.max_impact<=20.) & (self.max_touchdown_speed<=.35),
            lift_reward=(self.phase==1)*((s['front_z']-.045)/.075).clamp(0,1)*s['rear_fraction'],
            balance_reward=(self.phase==2)*balanced.float(),
            track_reward=(self.phase==2)*support*tracking_reward,
            land_reward=landing_progress,
            landing_time=(self.phase==3).float(),
            ground_quality=(self.phase==3)*s['four_fraction']*frame_match,
            descent_cost=(self.phase==3)*descent_cost,
            approach_reward=(self.phase==3)*torch.exp(-s['front_vertical_speed'].square()/.25**2)*(1-remaining),
            handoff_reward=(self.phase==4)*ground*frame_match,
            touchdown_cost=(self.phase==3)*s['front_contact']*torch.relu(-s['front_vertical_speed']-.25).square(),
            body_cost=self.active*s['body_history'],
            transition_track=((self.phase==1)|(self.phase==3))*.1*tracking_reward,
            smooth_reward=self.active/(1+(m['tilt_rate']/.8).square()))
        self.result=result;return result
