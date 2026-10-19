"""Tensor-only moving takeover state machine. No reference motions or model changes."""
import math
import torch


def slew(current, target, limit):
    return current + (target-current).clamp(-limit, limit)


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class MovingProgress:
    def __init__(self, n, device, dt, speed_limit=.3, yaw_limit=1., frame_range=(-1.,1.2)):
        self.n=n; self.device=device; self.dt=dt
        self.speed_limit=speed_limit; self.yaw_limit=yaw_limit; self.frame_range=frame_range
        self.window=math.ceil(1/dt)
        def zeros(*shape):return torch.zeros(*shape,device=device)
        self.external=zeros(n).bool(); self.active=zeros(n).bool(); self.age=zeros(n); self.since=zeros(n)
        self.ready=zeros(n); self.switch_at=zeros(n); self.next_command=zeros(n)
        self.raw=zeros(n,2); self.command=zeros(n,2)
        self.reference=zeros(n,2); self.raw_reference=zeros(n,2)
        self.yaw=zeros(n); self.raw_yaw=zeros(n); self.frame=zeros(n)
        self.peak=zeros(n); self.lifted=zeros(n).bool(); self.ever=zeros(n).bool()
        self.support_run=zeros(n); self.good=torch.zeros(self.window,n,device=device,dtype=torch.bool)
        self.cursor=0; self.last_g=zeros(n,3); self.mode=torch.arange(n,device=device)%6
        self.result={}

    def sample_commands(self, ids):
        # Retain straight, reverse, pure-turn and curved-motion cases in every batch.
        v=(2*torch.rand(len(ids),device=self.device)-1)*self.speed_limit
        w=(2*torch.rand(len(ids),device=self.device)-1)*self.yaw_limit
        mode=self.mode[ids]
        v=torch.where(mode==0,v.abs(),torch.where(mode==1,-v.abs(),v))
        w=torch.where((mode==0)|(mode==1),0.,w)
        v=torch.where((mode==2)|(mode==3),0.,v)
        w=torch.where(mode==2,w.abs(),torch.where(mode==3,-w.abs(),w))
        self.raw[ids]=torch.stack((v,w),-1)
        self.next_command[ids]=self.since[ids]+2.+2.*torch.rand(len(ids),device=self.device)

    def set_command(self, command, ids=None):
        """External (forward m/s, yaw rad/s) input; disable random resampling."""
        if ids is None:ids=torch.arange(self.n,device=self.device)
        command=torch.as_tensor(command,device=self.device,dtype=self.raw.dtype)
        if command.shape==(2,):command=command.expand(len(ids),2)
        if command.shape!=(len(ids),2) or not torch.isfinite(command).all():raise ValueError('Invalid remote command')
        if (command[:,0].abs()>self.speed_limit).any() or (command[:,1].abs()>self.yaw_limit).any():raise ValueError('Remote command exceeds configured envelope')
        self.raw[ids]=command;self.external[ids]=True

    def reset(self, ids, pos, heading):
        for x in (self.active,self.age,self.since,self.ready,self.peak,self.lifted,self.ever,self.support_run):x[ids]=0
        self.external[ids]=False; self.good[:,ids]=False; self.last_g[ids]=torch.tensor([0.,0.,-1.],device=self.device)
        self.frame[ids]=self.frame_range[0]+torch.rand(len(ids),device=self.device)*(self.frame_range[1]-self.frame_range[0])
        self.switch_at[ids]=2.+2.*torch.rand(len(ids),device=self.device)
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
        fixed_error=(s['q'][:,:3]-fixed).abs().amax(-1)
        tilt_rate=(s['g']-self.last_g).norm(dim=-1)/self.dt;self.last_g.copy_(s['g'])
        # Turning is allowed: no zero-speed requirement and no fixed world-yaw span.
        good=support & (raw_error[:,0].abs()<.12)&(raw_error[:,1].abs()<.3)&(raw_distance<.15)&(heading_error.abs()<.25)&(fixed_error<.05)&(s['g'][:,1].abs()<.15)&(tilt_rate<.8)
        self.good[self.cursor]=good;self.cursor=(self.cursor+1)%self.window
        held=self.good.all(0)&(self.since>=1.)
        fresh=held&~self.ever;self.ever |= held
        lift=((s['front_z']-.045)/.075).clamp(0,1)*s['rear']*(~s['other'])
        phi=(.25*(-s['g'][:,0]/.8).clamp(0,1)*s['rear']+lift+self.lifted.float())*self.active
        new_peak=torch.maximum(phi,self.peak);progress=(new_peak-self.peak)/self.dt;self.peak=new_peak
        finite=torch.isfinite(torch.cat((s['pos'],s['vel'],s['q']),-1)).all(-1)
        physical=(~finite)|(s['pos'][:,2]<.025)|(s['g'][:,2]>.2)|(s['vel'][:,:3].norm(dim=-1)>5)|(s['vel'][:,3:].norm(dim=-1)>30)
        warmup_failed=(~self.active)&(self.age>6.)
        failure=physical|warmup_failed|(self.active & (raw_distance>1.5))
        self.result=dict(active=self.active.clone(),switched=switch,support=support,held=held,fresh=fresh,lift_event=lift_event,
            failure=failure,warmup_failed=warmup_failed,physical_failure=physical,progress=progress,
            distance=distance,raw_distance=raw_distance,heading_error=heading_error,raw_error=raw_error,error=error,
            fixed_error=fixed_error,tilt_rate=tilt_rate,
            transition_error=self.active*(~self.lifted)*(raw_error[:,0].square()/.3**2+raw_error[:,1].square()).clamp(max=4),
            prepare=(~self.lifted)*self.active*(.05*(1-self.since/2).clamp(0,1)+.5*lift.square()),
            support_reward=support*(.2+.8*(self.support_run/.3).clamp(0,1)),
            tracking=support*torch.exp(-error[:,0].square()/.15**2-error[:,1].square()/.4**2),
            raw_tracking=support*torch.exp(-raw_error[:,0].square()/.2**2-raw_error[:,1].square()/.5**2),
            path=support*torch.exp(-raw_distance.square()/.2**2-heading_error.square()/.3**2),
            smooth=support/(1+(tilt_rate/.8).square()))
        return self.result
