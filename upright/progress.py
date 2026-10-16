"""Contact-driven lift/balance-and-return task state; no reference actions or poses."""
import math
import torch

GAMMA = .997

class Progress:
    def __init__(self, n, device, dt=.02):
        self.dt=dt; self.gamma=GAMMA**(dt/.02); self.window=math.ceil(1./dt); self.cursor=0
        self.enter_steps=math.ceil(.1/dt); self.leave_steps=math.ceil(.06/dt)
        self.origin=torch.zeros(n,2,device=device)
        self.phase=torch.zeros(n,device=device,dtype=torch.long)
        self.age=torch.zeros_like(self.phase)
        self.on=torch.zeros(n,device=device); self.off=self.on.clone()
        self.lifted=torch.zeros(n,device=device,dtype=torch.bool); self.ever=self.lifted.clone()
        self.previous_phi=self.on.clone(); self.hold=self.on.clone()
        # Gravity-vector and position-joint ranges reject real pose oscillation without
        # requiring every instantaneous joint velocity to be near zero.
        self.history=torch.zeros(self.window,n,10,device=device)
        self.valid=torch.zeros(self.window,n,device=device,dtype=torch.bool)
        self.near=self.valid.clone(); self.slow=self.valid.clone()
        self.contacts=torch.zeros(max(1,round(.06/dt)),n,device=device,dtype=torch.bool)
        self.result={}

    def reset(self, ids, s):
        self.origin[ids]=s['pos'][ids,:2]
        for v in (self.phase,self.age,self.on,self.off,self.lifted,self.ever,self.previous_phi,self.hold):v[ids]=0
        for v in (self.history,self.valid,self.near,self.slow,self.contacts):v[:,ids]=0

    def update(self, s, failure, fixed_targets):
        raw=s['rear'] & ~s['other'] & (s['front_z']>.12) & (s['g'][:,2]<-.25)
        self.contacts=torch.roll(self.contacts,1,0);self.contacts[0]=raw
        support=self.contacts.sum(0)>self.contacts.shape[0]//2
        self.on=torch.where(support,self.on+1,0.)
        self.off=torch.where(support,0.,self.off+1)
        enter=(self.phase==0)&(self.on>=self.enter_steps)
        lift_event=enter & ~self.lifted;self.lifted |= enter
        self.phase[enter]=1
        self.phase[self.off>=self.leave_steps]=0
        delta=s['pos'][:,:2]-self.origin;dist=delta.norm(dim=-1)
        target=-delta
        target=target*torch.clamp(.2/target.norm(dim=-1,keepdim=True).clamp_min(1e-6),max=1.)
        features=torch.cat((s['g'],s['heading'],s['q']),-1)
        i=self.cursor;self.history[i]=features;self.valid[i]=support
        fixed_error=(s['q'][:,:3]-fixed_targets).abs().amax(-1)
        self.near[i]=(dist<=.05)&(fixed_error<=.05)
        self.slow[i]=(s['vel'][:,:3].norm(dim=-1)<.1)
        self.age+=1;self.cursor=(i+1)%self.window
        span=self.history.amax(0)-self.history.amin(0)
        pose_stable=(span[:,:3].norm(dim=-1)<.05)&(span[:,3:5].norm(dim=-1)<.05)&(span[:,5:].amax(-1)<.03)
        full=self.age>=self.window
        balanced=full & self.valid.all(0)&self.slow.all(0)&pose_stable&~failure
        held=balanced & self.near.all(0)
        fresh=held & ~self.ever;self.ever |= held
        self.hold=torch.where(support & (dist<=.05) & (fixed_error<=.05) & (s['vel'][:,:3].norm(dim=-1)<.1),self.hold+self.dt,0.).clamp(max=1.)
        lift=((s['front_z']-.045)/.075).clamp(0,1)*s['rear'].float()*(~s['other']).float()
        quality=torch.exp(-dist.square()/.15**2)
        # Before wheel lift, the demonstrated mechanism first pitches the body.
        # Small bounded potential gives this preparatory motion credit, not success.
        prepare=.25*(-s['g'][:,0]/.8).clamp(0,1)*s['rear'].float()
        phi=prepare+lift+(self.phase==1).float()*quality
        phi=torch.where(failure,0.,phi)
        progress=(self.gamma*phi-self.previous_phi)/self.dt;self.previous_phi=phi
        # Broad feedback remains informative before a fully stable 1s window exists.
        velocity_quality=torch.exp(-(s['vel'][:,:2]-target).square().sum(-1)/.15**2)
        rotation_quality=torch.exp(-s['vel'][:,3:].square().sum(-1)/4.)
        active=support&~failure
        # Dense bridge before the phase event: sustained support earns more than
        # repeated brief bounces. Preparatory pitch shaping expires after 2 s.
        duration=(self.on*self.dt/.3).clamp(0,1)
        prep_decay=(1-self.age*self.dt/2.).clamp(0,1)
        self.result=dict(raw_support=raw,support=support,balanced=balanced,held=held,fresh=fresh,
            lift_event=lift_event,progress=progress,distance=dist,fixed_error=fixed_error,
            prepare_reward=(self.phase==0).float()*(.2*prepare*prep_decay+.5*lift.square())*(~failure).float(),
            support_reward=active.float()*(.2+.8*duration),balance=active.float()*velocity_quality*rotation_quality*(.1+.9*quality),
            home_hold=held.float(), phase=self.phase, pose_span=span, target_velocity=target)
        return self.result
