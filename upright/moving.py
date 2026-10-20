"""Tensor-only moving takeover state machine. No reference motions or model changes."""
import math
import torch


def slew(current, target, limit):
    return current + (target-current).clamp(-limit, limit)


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def projected_heading_rate(forward_axis, angular_velocity):
    """Derivative of atan2(body-forward world y, x), with world angular velocity."""
    dx=torch.cross(angular_velocity,forward_axis,dim=-1)
    return (forward_axis[:,0]*dx[:,1]-forward_axis[:,1]*dx[:,0])/forward_axis[:,:2].square().sum(-1).clamp_min(1e-6)


def heading_feedback_rewards(forward_error, heading_rate_error, distance, heading_error):
    # Independent terms keep angular feedback useful while translation is imperfect.
    tracking=.25*torch.exp(-forward_error.square()/.1**2)+.75*torch.exp(-heading_rate_error.square()/.12**2)
    path=.5*torch.exp(-distance.square()/.2**2)+.5*torch.exp(-heading_error.square()/.15**2)
    return tracking,path


class MovingProgress:
    def __init__(self, n, device, dt, speed_limit=.3, yaw_limit=1., frame_range=(-1.,1.2), seed=42, feedback_mode="command", turn_share=0.):
        self.n=n; self.device=device; self.dt=dt; self.seed=seed; self.feedback_mode=feedback_mode
        self.generation=torch.zeros(n,device=device,dtype=torch.long);self.command_count=self.generation.clone()
        self.speed_limit=speed_limit; self.yaw_limit=yaw_limit; self.frame_range=frame_range
        self.window=math.ceil(1/dt)
        def zeros(*shape):return torch.zeros(*shape,device=device)
        self.external=zeros(n).bool(); self.active=zeros(n).bool(); self.age=zeros(n); self.since=zeros(n)
        self.ready=zeros(n); self.switch_at=zeros(n); self.next_command=zeros(n)
        self.raw=zeros(n,2); self.command=zeros(n,2)
        self.reference=zeros(n,2); self.raw_reference=zeros(n,2)
        self.yaw=zeros(n); self.raw_yaw=zeros(n); self.frame=zeros(n)
        self.peak=zeros(n); self.lifted=zeros(n).bool(); self.ever=zeros(n).bool()
        self.support_run=zeros(n); self.heading_run=zeros(n)
        self.good=torch.zeros(self.window,n,device=device,dtype=torch.bool)
        self.cursor=0; self.last_g=zeros(n,3); self.mode=self.mode_pattern(n,device,turn_share)
        self.result={}

    @staticmethod
    def mode_pattern(n,device,turn_share):
        """Fixed per-env command class. turn_share raises the share of pure turns,
        because pure turning is where accumulated heading error is worst."""
        base=[0,1,4,5]
        turns=max(0,min(8,round(turn_share/(1-turn_share)*len(base)/2))) if 0<turn_share<1 else 0
        pattern=base+[2,3]*turns
        return (torch.arange(n,device=device)%len(pattern)).long().apply_(lambda i: pattern[i])

    def uniform(self, ids, stream):
        # Per-env counter-based streams: resetting another env cannot change the
        # held-out command sequence. Avoid shared RNG coupling during evaluation.
        x=(ids*1103515245+self.generation[ids]*12345+self.command_count[ids]*214013+self.seed*2531011+stream*134775813)&0x7fffffff
        x=((x^(x>>16))*1103515245+12345)&0x7fffffff
        x=((x^(x>>13))*214013+2531011)&0x7fffffff
        return x.to(torch.float32)/2147483648.

    def sample_commands(self, ids):
        # Retain straight, reverse, pure-turn and curved-motion cases in every batch.
        v=(2*self.uniform(ids,0)-1)*self.speed_limit
        w=(2*self.uniform(ids,1)-1)*self.yaw_limit
        mode=self.mode[ids]
        v=torch.where(mode==0,v.abs(),torch.where(mode==1,-v.abs(),v))
        w=torch.where((mode==0)|(mode==1),0.,w)
        v=torch.where((mode==2)|(mode==3),0.,v)
        w=torch.where(mode==2,w.abs(),torch.where(mode==3,-w.abs(),w))
        self.raw[ids]=torch.stack((v,w),-1)
        self.next_command[ids]=self.since[ids]+2.+2.*self.uniform(ids,2)
        self.command_count[ids]+=1

    def set_command(self, command, ids=None):
        """External (forward m/s, yaw rad/s) input; disable random resampling."""
        if ids is None:ids=torch.arange(self.n,device=self.device)
        command=torch.as_tensor(command,device=self.device,dtype=self.raw.dtype)
        if command.shape==(2,):command=command.expand(len(ids),2)
        if command.shape!=(len(ids),2) or not torch.isfinite(command).all():raise ValueError('Invalid remote command')
        if (command[:,0].abs()>self.speed_limit).any() or (command[:,1].abs()>self.yaw_limit).any():raise ValueError('Remote command exceeds configured envelope')
        self.raw[ids]=command;self.external[ids]=True

    def reset(self, ids, pos, heading):
        for x in (self.active,self.age,self.since,self.ready,self.peak,self.lifted,self.ever,self.support_run,self.heading_run):x[ids]=0
        self.generation[ids]+=1;self.command_count[ids]=0
        self.external[ids]=False; self.good[:,ids]=False; self.last_g[ids]=torch.tensor([0.,0.,-1.],device=self.device)
        self.frame[ids]=self.frame_range[0]+self.uniform(ids,3)*(self.frame_range[1]-self.frame_range[0])
        self.switch_at[ids]=2.+2.*self.uniform(ids,4)
        self.command[ids]=0
        self.reference[ids]=pos[ids,:2]; self.raw_reference[ids]=pos[ids,:2]
        self.yaw[ids]=heading[ids];self.raw_yaw[ids]=heading[ids]
        self.sample_commands(ids)

    def before_step(self):
        ids=torch.where(self.active & ~self.external & (self.since>=self.next_command))[0]
        if ids.numel():self.sample_commands(ids)
        # Keep accepting operator commands. During lift, use gentler command slew.
        transitional=self.active & (self.support_run<.3)
        acc=torch.where(transitional,.3,.6); yaw_acc=torch.where(transitional,.8,2.)
        self.command[:,0]=slew(self.command[:,0],self.raw[:,0],acc*self.dt)
        desired_yaw=self.raw[:,1].clamp(-self.yaw_limit,self.yaw_limit)
        lateral_limit=.4/self.command[:,0].abs().clamp_min(.05)
        desired_yaw=torch.maximum(torch.minimum(desired_yaw,lateral_limit),-lateral_limit)
        self.command[:,1]=slew(self.command[:,1],desired_yaw,yaw_acc*self.dt)
        for ref,yaw,command in ((self.reference,self.yaw,self.command),(self.raw_reference,self.raw_yaw,self.raw)):
            mid=yaw+.5*command[:,1]*self.dt
            delta=command[:,0,None]*torch.stack((mid.cos(),mid.sin()),-1)*self.dt
            ref.add_(delta*self.active[:,None]);yaw.add_(command[:,1]*self.dt*self.active)
            yaw.copy_(wrap(yaw))

    def update(self,s,fixed):
        self.age+=self.dt
        active_before=self.active.clone()
        # Gate only actual four-wheel contact and attained symmetric frame target.
        admissible=s['four'] & ((s['q'][:,3:]-self.frame[:,None]).abs().amax(-1)<.08)
        self.ready=torch.where(admissible,self.ready+self.dt,0.)
        switch=(~self.active)&(self.age>=self.switch_at)&(self.ready>=.12)
        self.active |= switch
        self.reference[switch]=s['pos'][switch,:2]; self.raw_reference[switch]=s['pos'][switch,:2]
        self.yaw[switch]=s['yaw'][switch];self.raw_yaw[switch]=s['yaw'][switch]
        # The first policy step inherits the measured motion; never request a stop.
        self.command[switch,0]=s['forward'][switch];self.command[switch,1]=s['vel'][switch,5]
        self.since+=active_before*self.dt
        support=self.active & s['rear'] & ~s['other'] & (s['front_z']>.12) & (s['g'][:,2]<-.25)
        self.support_run=torch.where(support,self.support_run+self.dt,0.)
        lift_event=(self.support_run>=.12)&~self.lifted;self.lifted |= lift_event
        raw_error=torch.stack((s['forward']-self.raw[:,0],s['vel'][:,5]-self.raw[:,1]),-1)
        error=torch.stack((s['forward']-self.command[:,0],s['vel'][:,5]-self.command[:,1]),-1)
        distance=(s['pos'][:,:2]-self.reference).norm(dim=-1)
        raw_distance=(s['pos'][:,:2]-self.raw_reference).norm(dim=-1)
        heading_error=wrap(s['yaw']-self.raw_yaw)
        feedback_target=pose_velocity_target(s['pos'][:,:2],s['yaw'],self.raw_reference,self.raw_yaw,self.command)
        feedback_error=torch.stack((s['forward'],s['vel'][:,5]),-1)-feedback_target
        tracking_reward=torch.exp(-error[:,0].square()/.15**2-error[:,1].square()/.4**2)
        path_reward=torch.exp(-raw_distance.square()/.2**2-heading_error.square()/.3**2)
        if self.feedback_mode=='pose_feedback':
            tracking_reward=torch.exp(-feedback_error[:,0].square()/.1**2-feedback_error[:,1].square()/.2**2)
        elif self.feedback_mode=='heading_feedback':
            tracking_reward,path_reward=heading_feedback_rewards(
                feedback_error[:,0],s['heading_rate']-feedback_target[:,1],raw_distance,heading_error)
        elif self.feedback_mode=='sustain':
            # Tight on what the success window actually measures: the 0.25 rad
            # heading budget is an integral, so the rate scale must be smaller than
            # the 0.4 rad/s the command-mode reward tolerated.
            tracking_reward=torch.exp(-feedback_error[:,0].square()/.1**2-feedback_error[:,1].square()/.15**2)
            path_reward=torch.exp(-raw_distance.square()/.2**2-heading_error.square()/.10**2)
        fixed_error=(s['q'][:,:3]-fixed).abs().amax(-1)
        tilt_rate=(s['g']-self.last_g).norm(dim=-1)/self.dt;self.last_g.copy_(s['g'])
        # Turning is allowed: no zero-speed requirement and no fixed world-yaw span.
        good=support & (raw_error[:,0].abs()<.12)&(raw_error[:,1].abs()<.3)&(raw_distance<.15)&(heading_error.abs()<.25)&(fixed_error<.05)&(s['g'][:,1].abs()<.15)&(tilt_rate<.8)
        # Dense credit for the criterion's own heading window: the hard gate needs
        # |heading error| inside it continuously, so reward the run length, not the instant.
        heading_ok=support&(heading_error.abs()<.25)
        self.heading_run=torch.where(heading_ok,self.heading_run+self.dt,0.)
        self.good[self.cursor]=good;self.cursor=(self.cursor+1)%self.window
        held=self.good.all(0)&(self.since>=1.)
        fresh=held&~self.ever;self.ever |= held
        lift=((s['front_z']-.045)/.075).clamp(0,1)*s['rear']*(~s['other'])
        prepare_pitch=(-s['g'][:,0]/.8).clamp(0,1)*s['rear']*(~s.get('body_contact',s['other']))
        # Smooth credit before the hard 12 cm success gate; never grant success
        # for partial lift. Tracking penalties stay nonzero during transition.
        transition_scale=.1+.9*(self.support_run/.3).clamp(0,1)
        phi=(.25*(-s['g'][:,0]/.8).clamp(0,1)*s['rear']+lift+self.lifted.float())*self.active
        new_peak=torch.maximum(phi,self.peak);progress=(new_peak-self.peak)/self.dt;self.peak=new_peak
        finite=torch.isfinite(torch.cat((s['pos'],s['vel'],s['q']),-1)).all(-1)
        physical=(~finite)|(s['pos'][:,2]<.025)|(s['g'][:,2]>.2)|(s['vel'][:,:3].norm(dim=-1)>5)|(s['vel'][:,3:].norm(dim=-1)>30)
        warmup_failed=(~self.active)&(self.age>6.)
        failure=physical|warmup_failed|(self.active & (raw_distance>1.5))
        self.result=dict(active=self.active.clone(),switched=switch,support=support,held=held,fresh=fresh,lift_event=lift_event,
            failure=failure,warmup_failed=warmup_failed,physical_failure=physical,progress=progress,
            distance=distance,raw_distance=raw_distance,heading_error=heading_error,raw_error=raw_error,error=error,
            fixed_error=fixed_error,tilt_rate=tilt_rate,feedback_target=feedback_target,lift_fraction=lift,transition_scale=transition_scale,
            transition_error=.1*self.active*(~self.lifted)*(raw_error[:,0].square()/.3**2+raw_error[:,1].square()).clamp(max=4),
            prepare=self.active*(.1*prepare_pitch*(1-self.since/2).clamp(0,1)+2.*lift),
            support_reward=support*(.2+.8*(self.support_run/.3).clamp(0,1)),
            heading_sustain=support*(self.heading_run/1.).clamp(0,1),
            tracking=support*tracking_reward,
            raw_tracking=support*torch.exp(-raw_error[:,0].square()/.2**2-raw_error[:,1].square()/.5**2),
            path=support*path_reward,
            smooth=support/(1+(tilt_rate/.8).square()))
        return self.result


def wheel_targets(actions,max_speed,mode):
    if mode=='independent':return actions[:,1:]*max_speed
    common=actions[:,1]*max_speed;differential=actions[:,2]*5.
    return torch.stack((common-differential,common+differential),-1).clamp(-max_speed,max_speed)


def pose_velocity_target(pos,yaw,reference,reference_yaw,command):
    """Bounded reward target only; never commands the actuators or changes success."""
    delta=reference-pos
    forward=delta[:,0]*yaw.cos()+delta[:,1]*yaw.sin()
    lateral=-delta[:,0]*yaw.sin()+delta[:,1]*yaw.cos()
    dv=forward.clamp(-.1,.1)
    # Reverse driving must steer the opposite way to remove the same lateral error.
    lateral_turn=.5*torch.atan2(lateral,torch.full_like(lateral,.25))*command[:,0].sign()
    dw=(1.5*wrap(reference_yaw-yaw)+lateral_turn).clamp(-.25,.25)
    return command+torch.stack((dv,dw),-1)
