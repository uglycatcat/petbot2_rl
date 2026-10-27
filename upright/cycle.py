"""Full moving upright cycle: rise, balance, land, then continuous ordinary-drive handoff."""
import torch
from .moving import MovingProgress, wrap


class CycleProgress(MovingProgress):
    def __init__(self,*args,tracking_shape="gaussian",handoff_alignment="before",transition_contract='legacy',**kwargs):
        if tracking_shape not in ("gaussian","cauchy"):raise ValueError("Invalid tracking reward shape")
        if handoff_alignment not in ('before','after'):raise ValueError('Invalid handoff alignment')
        self.tracking_shape=tracking_shape
        self.handoff_alignment=handoff_alignment
        assert transition_contract in ('legacy','linear_heading')
        self.transition_contract=transition_contract
        super().__init__(*args,**kwargs)
        z=lambda:torch.zeros(self.n,device=self.device)
        self.phase=z().long();self.requested_mode=z()+1;self.destination=z()
        self.phase_time=z();self.land_at=z();self.ground_run=z();self.balance_time=z()
        self.rise_position=torch.zeros(self.n,2,device=self.device)
        self.transition_offset=z();self.front_previous=z()+.045
        self.max_impact=z();self.max_touchdown_speed=z();self.last_front_speed=z();self.landed=z().bool();self.handoff=z().bool()
        self.balance_ever=z().bool();self.external_mode=z().bool();self.land_height=z()+.18
        self.track_sum=torch.zeros(self.n,2,device=self.device);self.track_count=z()
        self.track_good=z();self.balance_good=z();self.handoff_run=z();self.landing_potential=z()
        self.transition_speed=z();self.transition_heading=z();self.measured_heading=z()

    def before_step(self):
        if self.transition_contract=='legacy':
            return super().before_step()
        from .moving import slew
        ids=torch.where(self.active & ~self.external & (self.since>=self.next_command))[0]
        if ids.numel():self.sample_commands(ids)
        mask=(self.phase==1)|(self.phase==3)
        self.command[:,0]=slew(self.command[:,0],self.raw[:,0],.6*self.dt)
        self.command[mask,0]=self.transition_speed[mask]
        yaw=self.raw[:,1].clamp(-self.yaw_limit,self.yaw_limit)
        lateral=.4/self.command[:,0].abs().clamp_min(.05)
        yaw=torch.maximum(torch.minimum(yaw,lateral),-lateral)
        self.command[:,1]=slew(self.command[:,1],yaw,2.*self.dt)
        self.command[mask,1]=(3.*wrap(self.transition_heading-self.measured_heading))[mask].clamp(-1.5,1.5)
        queued=self.raw.clone();queued[mask]=self.command[mask]
        for ref,heading,command in ((self.reference,self.yaw,self.command),(self.raw_reference,self.raw_yaw,queued)):
            heading[mask]=self.transition_heading[mask]
            rate=torch.where(mask,0.,command[:,1])
            middle=heading+.5*rate*self.dt
            ref.add_(command[:,0,None]*torch.stack((middle.cos(),middle.sin()),-1)*self.dt*self.active[:,None])
            heading.copy_(wrap(heading+rate*self.dt*self.active))

    def sample_commands(self, ids):
        super().sample_commands(ids)
        # Six ordinary-drive groups: forward, reverse, left, right, curve, stop.
        self.raw[ids[self.mode[ids]==5]]=0.

    def reset(self,ids,pos,heading):
        super().reset(ids,pos,heading)
        for x in (self.phase,self.phase_time,self.ground_run,self.balance_time,self.transition_offset,
                  self.max_impact,self.max_touchdown_speed,self.last_front_speed,self.landed,self.handoff,self.balance_ever,self.external_mode,self.track_count,self.track_good,self.balance_good,self.handoff_run,self.landing_potential):x[ids]=0
        self.track_sum[ids]=0;self.requested_mode[ids]=1
        self.destination[ids]=self.frame_range[0]+self.uniform(ids,8)*(self.frame_range[1]-self.frame_range[0])
        self.land_at[ids]=5.+3.*self.uniform(ids,9)
        self.rise_position[ids]=pos[ids,:2];self.front_previous[ids]=.045;self.land_height[ids]=.18
        self.transition_speed[ids]=0.;self.transition_heading[ids]=heading[ids];self.measured_heading[ids]=heading[ids]

    def request_mode(self,mode,frame,ids=None):
        if ids is None:ids=torch.arange(self.n,device=self.device)
        mode=torch.as_tensor(mode,device=self.device,dtype=self.requested_mode.dtype).expand(len(ids))
        frame=torch.as_tensor(frame,device=self.device,dtype=self.destination.dtype).expand(len(ids))
        if not ((mode==0)|(mode==1)).all() or not torch.isfinite(frame).all():raise ValueError('Invalid mode request')
        if (frame<self.frame_range[0]).any() or (frame>self.frame_range[1]).any():raise ValueError('Driving frame outside envelope')
        # A requested landing is latched until stable handoff. Chattering cannot reset its deadline/reference.
        accept=(self.phase[ids]<3)
        selected=ids[accept];self.requested_mode[selected]=mode[accept];self.destination[selected]=frame[accept]
        self.external_mode[ids]=True

    def update(self,s,fixed):
        geometry=None
        self.measured_heading.copy_(s['yaw'])
        if self.transition_contract=='linear_heading':
            from .kinematics import wheel_geometry
            gap,rear_skew,geometry=wheel_geometry(s['g'],s['q'][:,3:])
            ready=s.copy();ready['four']=geometry
            # The inherited progress features must not leak simulated wheel
            # contact into the deployable stage/observation state.
            ready['rear']=(rear_skew<.025)&(s['g'][:,2]<.2)
            ready['other']=gap.amin(-1)<.01
            ready['body_contact']=s['g'][:,2]>.2
            ready['front_z']=gap.amin(-1)+.045
        else:ready=s
        # Base progress supplies warmup admission and command/filter machinery;
        # none of its reward/success definitions are used by this version.
        previous_support=self.support_run.clone();m=super().update(ready,fixed)
        switch=m['switched'];self.phase[switch]=1;self.phase_time[switch]=0
        self.rise_position[switch]=s['pos'][switch,:2]
        if self.transition_contract=='linear_heading':
            self.transition_speed[switch]=s['forward'][switch]
            self.transition_heading[switch]=s['yaw'][switch]
            self.command[switch,1]=0.
        self.phase_time+=self.active*self.dt
        support=self.active & (s['rear_fraction']>=.9) & ~s['other_history'] & (s['front_z']>.12) & (s['g'][:,2]<-.25)
        if self.transition_contract=='linear_heading':
            support=self.active & (gap.amin(-1)>.075) & (rear_skew<.025) & (s['g'][:,2]<-.25)
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
        self.requested_mode[automatic]=0
        landing=(self.phase==2)&(self.requested_mode==0)
        self.phase[landing]=3;self.phase_time[landing]=0;self.land_height[landing]=s['front_z'][landing].clamp_min(.12)
        if self.transition_contract=='linear_heading':
            self.transition_speed[landing]=s['forward'][landing]
            self.transition_heading[landing]=s['yaw'][landing]
            self.command[landing,0]=self.transition_speed[landing];self.command[landing,1]=0.
        ground=(s['four_fraction']>=.9)&~s['body_history']&(s['g'][:,2]<-.85)&(s['g'][:,1].abs()<.15)
        frame_error=(s['q'][:,3:]-self.destination[:,None]).abs().amax(-1)
        gentle=(s['front_vertical_speed'].abs()<.25)&(s['vel'][:,3:5].norm(dim=-1)<1.)
        if self.transition_contract=='linear_heading':
            ground=geometry
            gentle=m['tilt_rate']<1.2
        # Ordinary drive already slews the frame to its requested destination.
        # Start it only after stable physical landing; final completion still
        # requires aligned frame and one continuous second of stable support.
        aligned=(frame_error<.1) if self.handoff_alignment=='before' else torch.ones_like(ground)
        self.ground_run=torch.where((self.phase==3)&ground&gentle&aligned,self.ground_run+self.dt,0.)
        handoff=(self.phase==3)&(self.ground_run>=(.12 if self.transition_contract=='linear_heading' else .30))
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
        tracking_reward=tracking_score(error,tolerance,self.tracking_shape)
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
        landing_timeout=(self.phase==3)&(self.phase_time>4.)
        if self.transition_contract=='linear_heading':landing_timeout=torch.zeros_like(landing_timeout)
        failure=m['physical_failure']|m['warmup_failed']|((self.phase==1)&(self.phase_time>3.))|landing_timeout
        failure |= self.active & ((s['pos'][:,:2]-self.rise_position).norm(dim=-1)>getattr(self,'travel_limit',3.))
        self.max_impact=torch.maximum(self.max_impact,torch.where((self.phase==3),s['front_force_peak'],0.))
        touchdown_speed=torch.maximum((-s['front_vertical_speed']).clamp_min(0),(-self.last_front_speed).clamp_min(0))
        previous_impact_excess=torch.relu(self.max_touchdown_speed/.35-1).square()
        self.max_touchdown_speed=torch.maximum(self.max_touchdown_speed,torch.where((self.phase==3)&s['front_contact'],touchdown_speed,0.))
        impact_increment=(torch.relu(self.max_touchdown_speed/.35-1).square()-previous_impact_excess).clamp_min(0.)
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
            impact_increment=impact_increment,
            touchdown_cost=(self.phase==3)*s['front_contact']*torch.relu(-s['front_vertical_speed']-.25).square(),
            body_cost=self.active*s['body_history'],
            transition_track=((self.phase==1)|(self.phase==3))*.1*tracking_reward,
            smooth_reward=self.active/(1+(m['tilt_rate']/.8).square()))
        if self.transition_contract=='linear_heading':
            transition=(self.phase==1)|(self.phase==3)
            yaw_error=wrap(s['yaw']-self.transition_heading)
            rate_error=s['heading_rate']-self.command[:,1]
            result['transition_track']=transition*(
                .5/(1.+(yaw_error/.15).square())+.5/(1.+(rate_error/.5).square()))
            result['geometry_ground']=geometry
            result['wheel_gap']=gap
            result['transition_heading_error']=yaw_error
        self.result=result;return result


def tracking_score(error,tolerance,shape):
    """Positive, bounded feedback; Cauchy has the same half-height error as Gaussian."""
    x2=(error/(2*tolerance)).square()
    if shape=='gaussian':return torch.exp(-x2).mean(-1)
    if shape=='cauchy':return (1/(1+x2/0.6931471805599453)).mean(-1)
    raise ValueError('Invalid tracking reward shape')


def filter_differential(previous,desired,dt,tau,max_speed):
    """Smooth yaw actuation only, preserving the requested common wheel speed."""
    if tau==0:return desired
    if tau<0 or dt<=0:raise ValueError('Invalid differential filter timing')
    import math
    common=desired.mean(-1)
    differential=(desired[:,1]-desired[:,0])*.5
    old=(previous[:,1]-previous[:,0])*.5
    filtered=old+(1-math.exp(-dt/tau))*(differential-old)
    margin=(max_speed-common.abs()).clamp_min(0.)
    filtered=torch.maximum(torch.minimum(filtered,margin),-margin)
    return torch.stack((common-filtered,common+filtered),-1)
