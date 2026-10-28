"""Independent upright calls over the source wheel-target feasible domain.

This changes task orchestration, not MJCF physics. Old ContinuousProgress stays
available for the historical policy bundles and fixed experiment copies.
"""
import torch
from .continuous import ContinuousProgress
from .moving import slew, wrap
from .kinematics import wheel_geometry


CONTRACT = dict(version='independent_full_range_v2', wheel_radius=.03,
    axle_width=.10, wheel_target_limit=25., forward_slew=.6, yaw_slew=6.,
    command_hold_seconds=[3.,6.], warmup_timeout=24., rise_timeout=5.,
    lifted_gap=.025, returned_gap=.012, return_dwell=.12,
    low_range_sample_fraction=.20, boundary_sample_fraction=.25,
    interruption='release_to_ordinary_no_completion_credit_new_request_required',
    admission='physical_four_wheel_and_actual_motion_training_only',
    admission_response_cap=[.02,.06])

# A training data distribution, never a deployment or evaluation envelope.
# Fixed environment strata preserve step coverage even when hard calls end early.
TRAINING_MIXTURE = dict(version='stratified_v1', assignment='contiguous_equal_environment_strata',
    evaluation='disabled', levels=[
        dict(name='retention', motion=[.05,.15], frame=[-.2,.2]),
        dict(name='motion_medium', motion=[.15,.5], frame=[-.2,.2]),
        dict(name='motion_higher', motion=[.3,1.5], frame=[-.2,.2]),
        dict(name='frame_full', motion=[.05,.15], frame=[-1.,1.2]),
        dict(name='full', motion=[.75,15.], frame=[-1.,1.2])])


def project_command(command, wheel_limit=25.):
    """Radial projection onto |v|/r + |yaw|*b/(2r) <= wheel_limit."""
    demand=command[:,0].abs()/.03+command[:,1].abs()*(.10/.06)
    return command/(demand/wheel_limit).clamp_min(1.)[:,None]


def effective_request(progress):
    """The actor cannot act on queued remote requests during rise/landing."""
    transition=(progress.phase==1)|(progress.phase==3)
    return torch.where(transition[:,None],progress.command,progress.raw)


class FullRangeProgress(ContinuousProgress):
    full_range=True
    warmup_timeout=24.
    rise_timeout=5.

    def __init__(self,*args,training_mixture='off',**kwargs):
        super().__init__(*args,**kwargs)
        assert training_mixture in ('off','stratified_v1')
        self.training_mixture=training_mixture
        self.training_level=torch.arange(self.n,device=self.device)*5//self.n
        if training_mixture=='off':self.training_level[:]=-1
        z=lambda:torch.zeros(self.n,device=self.device)
        self.lift_seen=z().bool()
        self.return_dwell=z()
        self.interrupted=z().bool()
        self.interrupt_reason=z().long()
        self.exit_phase=z().long()
        self.requested=torch.zeros(self.n,2,device=self.device)

    def sampling_limits(self,ids):
        limits=self.raw.new_tensor([self.speed_limit,self.yaw_limit]).expand(len(ids),2)
        if self.training_mixture!='off':
            tier=self.raw.new_tensor([x['motion'] for x in TRAINING_MIXTURE['levels']])[self.training_level[ids]]
            limits=torch.minimum(limits,tier)
        return limits

    def sample_commands(self,ids):
        # Keep the six-class sequence; sample interior, boundary, and regression
        # amplitudes explicitly instead of a rectangular unreachable envelope.
        super().sample_commands(ids)
        limits=self.sampling_limits(ids)
        if self.training_mixture!='off':
            self.raw[ids]*=limits/self.raw.new_tensor([self.speed_limit,self.yaw_limit])
        command=project_command(self.raw[ids])
        choice=self.uniform(ids,15)
        low=choice<CONTRACT['low_range_sample_fraction']
        small=torch.minimum(limits,command.new_tensor([.05,.15]))
        command[low]=self.raw[ids][low]*small[low]/limits[low]
        boundary=(choice>=.20)&(choice<.45)&(self.mode[ids]!=5)
        demand=command[:,0].abs()/.75+command[:,1].abs()/15.
        # Do not expand diagnostic evaluation beyond its declared envelope.
        factor=torch.minimum((.85+.15*self.uniform(ids,16))/demand.clamp_min(1e-8),
            (limits/command.abs().clamp_min(1e-8)).amin(-1))
        command[boundary]*=factor[boundary,None]
        self.raw[ids]=project_command(command)
        ramp=((self.raw[ids]-self.command[ids]).abs()/command.new_tensor([.6,6.])).amax(-1)
        self.next_command[ids]=self.since[ids]+ramp+3.+3.*self.uniform(ids,2)

    def set_command(self,command,ids=None):
        super().set_command(command,ids)
        selected=slice(None) if ids is None else ids
        self.requested[selected]=self.raw[selected]
        self.raw[selected]=project_command(self.raw[selected])

    def reset(self,ids,pos,heading):
        super().reset(ids,pos,heading)
        if self.training_mixture!='off':
            bounds=self.raw.new_tensor([x['frame'] for x in TRAINING_MIXTURE['levels']])[self.training_level[ids]]
            lo=bounds[:,0].clamp_min(self.frame_range[0]);hi=bounds[:,1].clamp_max(self.frame_range[1])
            assert (hi>=lo).all(), 'Training stratum outside frame envelope'
            span=max(self.frame_range[1]-self.frame_range[0],1e-8)
            for value in (self.frame,self.destination):
                value[ids]=lo+(value[ids]-self.frame_range[0])/span*(hi-lo)
        for value in (self.lift_seen,self.return_dwell,self.interrupted,self.interrupt_reason,self.exit_phase):
            value[ids]=0
        self.requested[ids]=self.raw[ids]
        ramp=(self.raw[ids].abs()/self.raw.new_tensor([.6,6.])).amax(-1)
        self.switch_at[ids]=ramp+2.+2.*self.uniform(ids,4)

    def before_step(self):
        ids=torch.where(self.active & ~self.external & (self.since>=self.next_command))[0]
        if ids.numel():self.sample_commands(ids)
        self.requested[~self.external]=self.raw[~self.external]
        self.command[:,0]=slew(self.command[:,0],self.raw[:,0],.6*self.dt)
        self.command[:,1]=slew(self.command[:,1],self.raw[:,1],6.*self.dt)
        self.command.copy_(project_command(self.command))
        transition=(self.phase==1)|(self.phase==3)
        self.command[transition,0]=self.transition_speed[transition]
        self.command[transition,1]=(3.*wrap(self.transition_heading-self.measured_heading))[transition].clamp(-15.,15.)
        queued=effective_request(self)
        for ref,heading,command in ((self.reference,self.yaw,self.command),(self.raw_reference,self.raw_yaw,queued)):
            heading[transition]=self.transition_heading[transition]
            rate=torch.where(transition,0.,command[:,1])
            middle=heading+.5*rate*self.dt
            ref.add_(command[:,0,None]*torch.stack((middle.cos(),middle.sin()),-1)*self.dt*self.active[:,None])
            heading.copy_(wrap(heading+rate*self.dt*self.active))

    def update(self,s,fixed):
        gap,rear_skew,ground=wheel_geometry(s['g'],s['q'][:,3:])
        previous_phase=self.phase.clone()
        self.lift_seen |= self.active & (gap.amin(-1)>.025) & (s['g'][:,2]<-.25)
        # Training must actually be moving; operator admission is governed by
        # posture and a new request, never exact tracking of an old command.
        deadband=self.raw.new_tensor([.005,.015])
        response=self.motion*self.raw.sign()
        nonzero=self.raw.abs()>deadband
        response_floor=torch.minimum(.25*self.raw.abs(),self.raw.new_tensor(CONTRACT['admission_response_cap']))
        # Saturation or cross-axis drift is not a reason to discard a genuinely
        # moving takeover. Only a stopped request asks for settled preparation.
        responding=(nonzero & (response>=response_floor)).any(-1)
        stopped=(self.motion.abs()<self.raw.new_tensor([.05,.15])).all(-1)
        moving=torch.where(nonzero.any(-1),responding,stopped)
        ready=s.copy()
        ready['admission_ready']=self.active | self.external_mode | moving
        result=super().update(ready,fixed)
        returned=((previous_phase==1)|(previous_phase==2)) & self.lift_seen & (gap.amin(-1)<.012) & (rear_skew<.025) & (s['g'][:,2]<-.5)
        self.return_dwell=torch.where(returned,self.return_dwell+self.dt,0.)
        accidental=self.return_dwell+1e-6>=.12
        abnormal=self.active & result['failure']
        exit_now=(accidental|abnormal)&~self.interrupted
        self.exit_phase[exit_now]=previous_phase[exit_now]
        self.interrupt_reason[exit_now]=torch.where(accidental[exit_now],1,2)
        self.interrupted |= exit_now
        # In training failure terminates this call. Deployment continues ordinary
        # control from the same physical state, with no upright auto-retry.
        self.phase[exit_now]=4
        self.phase_time[exit_now]=0.
        self.requested_mode[exit_now]=0.
        self.external_mode[exit_now]=True
        self.handoff[self.interrupted]=False
        result['failure'] |= self.interrupted
        result['completed'] &= ~self.interrupted
        result['qualified'] &= ~self.interrupted
        result['entered_handoff'] &= ~self.interrupted
        result['handoff_reward'][self.interrupted]=0.
        result['failure_code'] |= accidental.long()*(1<<13)
        result.update(interrupted=self.interrupted.clone(),interrupt_reason=self.interrupt_reason.clone(),
            exit_phase=self.exit_phase.clone(),lift_seen=self.lift_seen.clone(),
            training_level=self.training_level.clone(),
            requested_command=self.requested.clone(),effective_request=effective_request(self),
            actual_admission=moving.clone(),geometry_ground=ground,wheel_gap=gap)
        self.result=result
        return result
